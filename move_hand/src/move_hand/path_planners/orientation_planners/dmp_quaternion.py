from __future__ import division, print_function

from typing import Tuple, List, Union

import numpy as np
import quaternion  # add-on numpy quaternion type (https://github.com/moble/quaternion)

class CanonicalSystem(object):
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.step_vectorized = np.vectorize(self.step, otypes=[float])
        self.reset()
        self.x = 1.0

    def step(self, dt: float, tau: float):
        """
        Solve the canonical system at next time step t+dt.

        Parameters
        ----------
        dt : float
            Time step.
        tau : float
            Temporal scaling factor.
        """
        x = self.x
        self.x += -self.alpha * x / tau * dt  # dx/dt = alpha * x / tau
        return x

    def rollout(self, t: List[float], tau: Union[float, List[float]]):
        """
        Solve the canonical system.

        Parameters
        ----------
        t : array_like
            Time points for which to evaluate the integral.
        tau : array_like
            Temporal scaling factor (scalar constant or same length as t).
        """
        self.reset()
        return self.step_vectorized(np.gradient(t), tau)

    def reset(self):
        self.x = 1.0

class QuaternionDMP:
    """
    [1] A. Ude, B. Nemec, T. Petric, and J. Morimoto, "Orientation in Cartesian
    space dynamic movement primitives", in 2014 IEEE International Conference on
    Robotics and Automation (ICRA), 2014, no. 3, pp 2997-3004.
    """

    def __init__(self, n_bfs=10, alpha: float = 48.0, beta: float = None, cs_alpha=None, cs=None, roto_dilatation=False):
        """
        Parameters
        ----------
        n_bfs : int
            Number of basis functions.
        alpha : float
            Filter constant.
        beta : float
            Filter constant.
        """

        self.n_bfs = n_bfs
        self.alpha = alpha
        self.beta = beta if beta is not None else self.alpha / 4
        self.cs = cs if cs is not None else CanonicalSystem(alpha=cs_alpha if cs_alpha is not None else self.alpha/2)

        # Centres of the Gaussian basis functions
        self.c = np.exp(-self.cs.alpha * np.linspace(0, 1, self.n_bfs))

        # Variance of the Gaussian basis functions
        self.h = 1.0 / np.gradient(self.c)**2
        # self.h = 0.0025

        # Scaling factor
        self.Do = np.identity(3)

        # Initially weights are zero (no forcing term)
        self.w = np.zeros((3, self.n_bfs))

        # Initial- and goal orientations
        self._q0 = quaternion.one
        self._go = quaternion.one

        self._q0_train = quaternion.one
        self._go_train = quaternion.one

        self._R_fx = np.identity(3)

        # Reset
        self.q = self._q0.copy()
        self.omega = np.zeros(3)
        self.d_omega = np.zeros(3)
        self.train_quats = None
        self.train_omega = None
        self.train_d_omega = None

        self._roto_dilatation = roto_dilatation

    def plan_path(self, start_quat, goal_quat, num_points, dt):
        # Calculate time vector
        time = (num_points-1)*dt
        tt = np.linspace(0, time, num_points-1)

        # Set the initial and goal positions
        self.q0 = quaternion.from_float_array(np.roll(start_quat, 1, axis=-1))
        self.go = quaternion.from_float_array(np.roll(goal_quat, 1, axis=-1))
        self.reset()
        
        # Compute the cartesian DMP trajectory
        quat, _, _ = self.rollout(tt, time)
        
        # Convert quaternion to rotation matrix
        quat = quaternion.as_float_array(quat)
        quat = np.roll(quat, -1, axis=-1) # To convert from w,x,y,z to x,y,z,w
        quat = np.insert(quat, 0, start_quat, 0)
        
        return quat.T
    
    def step(self, x, dt, tau, torque_disturbance=np.array([0, 0, 0])) -> Tuple[quaternion.quaternion, np.ndarray, np.ndarray]:
        def fo(xj):
            psi = np.exp(-self.h * (xj - self.c)**2)
            return self.Do.dot(self.w.dot(psi) / psi.sum() * xj)

        # DMP system acceleration
        self.d_omega = (self.alpha * (self.beta * 2 * np.log(self._go * self.q.conjugate()).vec -
                                      tau * self.omega) + self._R_fx @ fo(x) + torque_disturbance) / tau ** 2

        # Integrate rotational acceleration
        self.omega += self.d_omega * dt

        # Integrate rotational velocity (to obtain quaternion)
        self.q = np.exp(dt / 2 * np.quaternion(0, *self.omega)) * self.q

        return self.q, self.omega, self.d_omega

    def rollout(self, ts, tau):
        self.reset()

        if np.isscalar(tau):
            tau = np.full_like(ts, tau)

        x = self.cs.rollout(ts, tau)  # Integrate canonical system
        dt = np.gradient(ts) # Differential time vector

        n_steps = len(ts)
        q = np.empty((n_steps,), dtype=np.quaternion)
        omega = np.empty((n_steps, 3))
        d_omega = np.empty((n_steps, 3))

        for i in range(n_steps):
            q[i], omega[i], d_omega[i] = self.step(x[i], dt[i], tau[i])

        return q, omega, d_omega

    @property
    def go(self):
        return self._go

    @go.setter
    def go(self, value):
        self._go = value

    @property
    def q0(self):
        return self._q0

    @q0.setter
    def q0(self, value):
        self._q0 = value
