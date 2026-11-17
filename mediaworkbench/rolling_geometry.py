"""Body-clock gyro integration and inverse rolling-shutter projection.

Camera axes: right, down, forward. A pose actively maps camera to world.
Each gyro sample holds a body angular velocity until the next timestamp, so
successive poses compose on the right (in the moving camera frame).

A sensor row is exposed at ``center + sign*((y+.5)/height-.5)*readout``.
The inverse map therefore cannot use the output row's time: it must solve for
the source row whose own exposure time images the requested world ray there.
"""
import numpy as np
from scipy.spatial.transform import Rotation

# Scratch arrays stay bounded independently of the sensor size.
_CHUNK_PIXELS = 32768
_MAX_SOLVE_ITERATIONS = 20
_SOLVE_TOLERANCE = 1e-6  # source pixels: far below one code sample


class Trajectory:
    def __init__(self, samples):
        self.times = np.asarray([s['at_ms'] for s in samples], dtype=float)
        self.rates = np.asarray([s['radians_per_second'] for s in samples], dtype=float)
        poses = [Rotation.identity()]
        for index in range(len(self.times)-1):
            increment = self.rates[index]*((self.times[index+1]-self.times[index])/1000)
            # Body-frame rates compose on the right; world-frame vector sums
            # are only valid while every sample keeps a single fixed axis.
            poses.append(poses[-1]*Rotation.from_rotvec(increment))
        self.poses = Rotation.concatenate(poses)
        self.matrices = self.poses.as_matrix()

    def at(self, times):
        times = np.asarray(times, dtype=float).reshape(-1)
        if np.any(times < self.times[0]) or np.any(times > self.times[-1]):
            raise ValueError('gyro does not cover sensor exposure')
        index = np.clip(np.searchsorted(self.times, times, side='right')-1,
                        0, len(self.times)-2)
        elapsed = (times-self.times[index])/1000
        return self.poses[index]*Rotation.from_rotvec(self.rates[index]*elapsed[:, None])

    def camera_rays(self, world, times, derivative=False):
        """Map world directions back to the camera exposed at each time."""
        times = np.asarray(times, dtype=float)
        if np.any(times < self.times[0]) or np.any(times > self.times[-1]):
            raise ValueError('gyro does not cover sensor exposure')
        index = np.clip(np.searchsorted(self.times, times, side='right')-1,
                        0, len(self.times)-2)
        elapsed = (times-self.times[index])/1000
        rate = self.rates[index]
        # R(t) = C_i Exp(omega_i elapsed), so R(t)^-1 = Exp(-...) C_i^T.
        base = np.einsum('...ji,...j->...i', self.matrices[index], world)
        rays = _rotate(base, -(rate*elapsed[..., None]))
        if not derivative:
            return rays
        # d/dt Exp(a t) v = a x Exp(a t) v, with a = -omega. The exposure
        # clock is milliseconds, matching the row-time expression above.
        return rays, np.cross(-rate/1000, rays)


def _rotate(vectors, rotvec):
    """Apply Exp(rotvec) row-wise with Rodrigues' formula."""
    squared = np.einsum('...i,...i->...', rotvec, rotvec)
    angle = np.sqrt(squared)
    small = angle < 1e-6
    safe_angle = np.where(small, 1.0, angle)
    safe_squared = np.where(small, 1.0, squared)
    sine = np.where(small, 1.0-squared/6.0, np.sin(safe_angle)/safe_angle)
    cosine = np.where(small, 1.0-squared/2.0, np.cos(safe_angle))
    half = np.where(small, .5-squared/24.0, (1.0-cosine)/safe_squared)
    cross = np.cross(rotvec, vectors)
    along = np.einsum('...i,...i->...', rotvec, vectors)
    return (vectors*cosine[..., None] + cross*sine[..., None]
            + rotvec*(along*half)[..., None])


def _chunk_rows(width):
    return max(1, _CHUNK_PIXELS//max(width, 1))


def _check_support(x, y, width, height):
    tolerance = 1e-7
    if (not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
            or np.min(x) < -tolerance or np.max(x) > width-1+tolerance
            or np.min(y) < -tolerance or np.max(y) > height-1+tolerance):
        raise ValueError('rolling crop exposes pixels outside original sensor')
    return np.clip(x, 0, width-1), np.clip(y, 0, height-1)


def mapping_rows(trajectory, center_ms, width, height, horizontal_fov, zoom,
                 readout_ms, direction, row_start, row_stop):
    """Inverse map for one band of output rows, at source pixel coordinates."""
    if direction == 'top-to-bottom':
        sign = 1.0
    elif direction == 'bottom-to-top':
        sign = -1.0
    else:
        raise ValueError('invalid sensor scan direction')
    focal = width/(2*np.tan(np.deg2rad(horizontal_fov)/2))
    rows = row_stop-row_start
    xx = np.arange(width, dtype=float)[None, :]
    yy = np.arange(row_start, row_stop, dtype=float)[:, None]
    x_out = xx+.5-width/2
    y_out = yy+.5-height/2
    shape = (rows, width)
    rx = np.broadcast_to(x_out/(focal*zoom), shape)
    ry = np.broadcast_to(y_out/(focal*zoom), shape)
    camera = np.stack((rx, ry, np.ones(shape)), axis=-1)
    # The output pixel centre is the global-shutter ray at the frame midpoint.
    world = camera @ trajectory.at([center_ms])[0].as_matrix().T

    if readout_ms == 0:
        # Every row is exposed at the midpoint: the map is a pure centre crop.
        if zoom == 1:
            # Keep the identity map exact so lossless delivery can return the
            # original bytes rather than a float-noise-resampled copy.
            x = np.broadcast_to(xx, shape).astype(float)
            y = np.broadcast_to(yy, shape).astype(float)
        else:
            x = np.broadcast_to(x_out/zoom+width/2-.5, shape)
            y = np.broadcast_to(y_out/zoom+height/2-.5, shape)
        return _check_support(np.array(x), np.array(y), width, height)

    # The first guess is the readout-free crop position. Each iteration solves
    # y = project(row time(y)) by Newton's method; a solution that leaves the
    # sensor is rejected rather than clamped or extrapolated.
    source_y = np.broadcast_to(y_out/zoom+height/2-.5, shape).copy()
    converged = False
    for _ in range(_MAX_SOLVE_ITERATIONS):
        # Rows outside the sensor have no exposure time. Probing at the
        # nearest real row keeps the gyro-span check meaningful while the
        # support check below still refuses the untrustworthy position.
        probe = np.clip(source_y, 0, height-1)
        exposed = center_ms+sign*((probe+.5)/height-.5)*readout_ms
        if (np.any(exposed < trajectory.times[0])
                or np.any(exposed > trajectory.times[-1])):
            raise ValueError('gyro does not cover sensor exposure')
        rays, derivative = trajectory.camera_rays(world, exposed, True)
        if np.any(rays[..., 2] <= 0):
            raise ValueError('rolling projection crosses camera horizon')
        projected = focal*rays[..., 1]/rays[..., 2]+height/2-.5
        step = np.max(np.abs(projected-source_y))
        if step <= _SOLVE_TOLERANCE:
            converged = True
            break
        # Newton on f(y) = project(row time(y)) - y, whose slope is
        # (sign*readout/height)*d(project)/dt - 1. The fixed-point step is the
        # fallback if that slope is degenerate.
        slope = sign*(readout_ms/height)*focal*(
            derivative[..., 1]*rays[..., 2]-rays[..., 1]*derivative[..., 2]
        )/(rays[..., 2]**2)
        usable = np.abs(slope-1.0) > 1e-6
        newton = source_y-(projected-source_y)/np.where(usable, slope-1.0, 1.0)
        source_y = np.where(usable, newton, projected)
    if not converged:
        raise ValueError('rolling projection has no trustworthy source row')
    # The accepted rays were evaluated at the returned row's own exposure
    # time, so that row images the requested world direction there.
    x = focal*rays[..., 0]/rays[..., 2]+width/2-.5
    y = projected
    return _check_support(x, y, width, height)


def mapping(trajectory, center_ms, width, height, horizontal_fov, zoom,
            readout_ms, direction):
    """Full inverse map as two (height, width) arrays of source coordinates."""
    # Sensor exposures run up to half a readout either side of the midpoint;
    # a gyro trace that stops at the nominal cut edge cannot describe them.
    reach = abs(readout_ms)*(height-.5)/height
    if center_ms-reach < trajectory.times[0] or center_ms+reach > trajectory.times[-1]:
        raise ValueError('gyro does not cover sensor exposure')
    blocks = [(start, min(start+_chunk_rows(width), height))
              for start in range(0, height, _chunk_rows(width))]
    xs, ys = [], []
    for start, stop in blocks:
        x, y = mapping_rows(trajectory, center_ms, width, height, horizontal_fov,
                            zoom, readout_ms, direction, start, stop)
        xs.append(x)
        ys.append(y)
    return np.vstack(xs), np.vstack(ys)
