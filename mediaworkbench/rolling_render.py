"""Frame-bounded linear-light rolling shutter correction to lossless FFV1."""
import os
from pathlib import Path
import tempfile
import numpy as np
from . import rolling_plan, rolling_geometry, media_sources, stitch_render, stitch_signal, stitch_plan
from .assets import fingerprint
from .tools import run


def sample(image, uv):
    x, y = uv
    h, w = image.shape[:2]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
    dx, dy = (x-x0)[..., None], (y-y0)[..., None]
    # Only the four supporting texels are decoded to scene-linear light, then
    # interpolated: averaging encoded code values would create dark fringes
    # beside high-contrast grid lines.
    top = stitch_signal.linear(image[y0, x0])*(1-dx)+stitch_signal.linear(image[y0, x1])*dx
    bottom = stitch_signal.linear(image[y1, x0])*(1-dx)+stitch_signal.linear(image[y1, x1])*dx
    return stitch_signal.encoded(top*(1-dy)+bottom*dy)


# Bands keep the float working set bounded for long programmes.
_BAND_ROWS = 24


def resample(image, x, y):
    h, w = image.shape[:2]
    result = np.empty_like(image)
    for start in range(0, h, _BAND_ROWS):
        stop = min(start+_BAND_ROWS, h)
        result[start:stop] = sample(image, (x[start:stop], y[start:stop]))
    return result


def corrected(image, trajectory, plan, frame):
    """Global-shutter picture at this frame's original-recording midpoint."""
    h, w = image.shape[:2]
    center_ms = plan['start_ms']+(frame+.5)*40
    if plan['readout_ms'] == 0 and plan['zoom'] == 1:
        # Every row is exposed at the midpoint and no crop is applied, so the
        # output is the original frame at its own pixel centres, byte for byte.
        return image
    x, y = rolling_geometry.mapping(trajectory, center_ms, w, h, plan['horizontal_fov'],
        plan['zoom'], plan['readout_ms'], plan['direction'])
    return resample(image, x, y)


def render(conn, identity, destination, at_frame=0, frame_count=None):
    p = rolling_plan.get(conn, identity)
    frame_count = p['frames']-at_frame if frame_count is None else frame_count
    if (not stitch_plan.integer(at_frame, 0, p['frames']-1)
            or not stitch_plan.integer(frame_count, 1, p['frames']-at_frame)):
        raise ValueError('invalid rolling preview range')
    target = Path(destination).expanduser().absolute()
    target = target.parent.resolve()/target.name
    if os.path.lexists(target) or not target.parent.is_dir():
        raise ValueError('rolling destination exists or parent missing')
    source = media_sources.select(conn, p['source'])
    w, h = source['width'], source['height']
    trajectory = rolling_geometry.Trajectory(p['gyro'])
    with tempfile.TemporaryDirectory(prefix='.rolling-', dir=target.parent) as directory:
        root = Path(directory)
        spool, raw = root/'source.rgb', root/'corrected.rgb'
        stitch_render.decode(source, p['start_ms'], p['frames'], spool)
        size = w*h*3
        with spool.open('rb') as handle, raw.open('wb') as out:
            # A preview slice reads the same spooled frames a complete
            # delivery would use, at the same original-recording times.
            handle.seek(at_frame*size)
            for frame in range(at_frame, at_frame+frame_count):
                data = handle.read(size)
                if len(data) != size:
                    raise ValueError('truncated rolling source spool')
                image = np.frombuffer(data, dtype='uint8').reshape(h, w, 3)
                out.write(corrected(image, trajectory, p, frame).tobytes())
        result = root/'corrected.mkv'
        run('ffmpeg', ['-v','error','-y','-threads','1','-filter_threads','1',
            '-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','25','-i',str(raw),
            '-an','-c:v','ffv1','-level','3','-pix_fmt','gbrp',str(result)])
        media_sources.select(conn, source)
        checksum = fingerprint(result)
        with result.open('rb') as handle:
            os.fsync(handle.fileno())
        os.link(result, target)
    return dict(plan_id=identity, path=str(target), at_frame=at_frame,
                frames=frame_count, sha256=checksum)
