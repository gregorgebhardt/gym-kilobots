import abc

import numpy as np
import Box2D


class Body:
    _density = 2
    _friction = 0.01
    _restitution = 0.0

    _linear_damping = 8
    _angular_damping = 8

    def __init__(self, world: Box2D.b2World, position=None, orientation=None):
        if self.__class__ == Body:
            raise NotImplementedError('Abstract class Body cannot be instantiated.')
        self._body_color = np.array((93, 133, 195))
        self._highlight_color = np.array((238, 80, 62))

        if position is None:
            position = [.0, .0]

        if orientation is None:
            orientation = .0

        self._world = world
        self._body = world.CreateDynamicBody(
            position=Box2D.b2Vec2(*position),
            angle=orientation,
            linearDamping=self._linear_damping,
            angularDamping=self._angular_damping)
        self._body.linearVelocity = Box2D.b2Vec2(*[.0, .0])
        self._body.angularVelocity = .0

    def __del__(self):
        self._world.DestroyBody(self._body)

    def get_position(self):
        return (*self._body.position,)

    def get_orientation(self):
        return self._body.angle

    def get_pose(self):
        return (*self._body.position, self._body.angle)

    def get_state(self):
        return (*self._body.position, self._body.angle)

    def get_local_point(self, point):
        return (*self._body.GetLocalPoint(point),)

    def get_local_orientation(self, angle):
        return angle - self._body.angle

    def get_local_pose(self, pose):
        return (*self.get_local_point(pose[:2]), self.get_local_orientation(pose[2]))

    @abc.abstractmethod
    def draw(self, viewer):
        raise NotImplementedError('The draw method needs to be implemented by the subclass of Body.')

    @abc.abstractmethod
    def plot(self, axes, **kwargs):
        raise NotImplementedError('The plot method needs to be implemented by the subclass of Body.')


class Quad(Body):
    def __init__(self, width, height, **kwargs):
        super().__init__(**kwargs)

        self._width = width
        self._height = height

        self._fixture = self._body.CreatePolygonFixture(
            box=Box2D.b2Vec2(self._width/2, self._height/2),
            density=self._density,
            friction=self._friction,
            restitution=self._restitution,
            radius=.001)

    def plot(self, axes, **kwargs):
        from matplotlib.patches import Rectangle
        defaults = dict(fill=True, edgecolor='#929591', facecolor='#d8dcd6')
        for k in defaults:
            if k not in kwargs:
                kwargs[k] = defaults[k]

        if 'alpha' in kwargs:
            from matplotlib.colors import to_rgba
            kwargs['edgecolor'] = to_rgba(kwargs['edgecolor'], kwargs['alpha'])
            kwargs['facecolor'] = to_rgba(kwargs['facecolor'], kwargs['alpha'])

        x, y, theta = self.get_pose()
        axes.add_patch(Rectangle(xy=(x-self._width/2, y-self._height/2), angle=theta,
                                 width=self._width, height=self._height, **kwargs))


class CornerQuad(Quad):
    def draw(self, viewer):
        super(CornerQuad, self).draw(viewer)

        # h = viewer.height
        # s = self.scale_sim_to_vis

        vertices = [self._body.transform * v for v in self._fixture.shape.vertices]
        # vertices = [(s * x, h - s * y) for (x, y) in vertices]

        viewer.draw_polygon(vertices[0:3], filled=True, color=self._highlight_color)

    def plot(self, axes, **kwargs):
        from matplotlib.patches import Polygon
        defaults = dict(highlight_fill=True, highlight_facecolor='#d8dcd6')
        for k in defaults:
            if k not in kwargs:
                kwargs[k] = defaults[k]

        if 'alpha' in kwargs:
            from matplotlib.colors import to_rgba
            kwargs['highlight_facecolor'] = to_rgba(kwargs['highlight_facecolor'], kwargs['alpha'])

        super(CornerQuad, self).plot(axes, **kwargs)
        vertices = [self._body.transform * v for v in self._fixture.shape.vertices]
        axes.add_patch(Polygon(xy=np.array(vertices[0:3]), fill=kwargs['highlight_fill'],
                               facecolor=kwargs['highlight_facecolor']))


class Circle(Body):
    def __init__(self, radius, **kwargs):
        super().__init__(**kwargs)

        self._radius = radius

        self._fixture = self._body.CreateCircleFixture(
            radius=self._radius,
            density=self._density,
            friction=self._friction,
            restitution=self._restitution
        )

    def draw(self, viewer):
        viewer.draw_circle(position=self._body.position, radius=self._radius, color=self._body_color)

    def get_radius(self):
        return self._radius

    def plot(self, axes, **kwargs):
        from matplotlib.patches import Circle
        defaults = dict(fill=True, edgecolor='#929591', facecolor='#d8dcd6')
        for k in defaults:
            if k not in kwargs:
                kwargs[k] = defaults[k]

        axes.add_patch(Circle(xy=self.get_position(), radius=self.get_radius(), **kwargs))


class Polygon(Body):
    def __init__(self, width: float, height: float, **kwargs):
        super().__init__(**kwargs)

        self.width = width
        self.height = height

        # TODO: right now this assumes that all subpolygons have the same number of edges
        # TODO: rewrite such that arbitrary subpolygons can be used here
        vertices = self._shape_vertices()

        v_size = np.amax(vertices, (0, 1)) - np.amin(vertices, (0, 1))
        vertices /= v_size
        vertices *= np.array((width, height))

        # compute centroid
        d = vertices * np.roll(np.roll(vertices, 1, axis=1), 1, axis=2)
        d = d[..., 0] - d[..., 1]
        A = .5 * d.sum(axis=1)
        centroids = np.sum((vertices + np.roll(vertices, 1, axis=1)) * d[..., None], axis=1) / (6*A[:, None])

        centroid = np.abs(A[None, :]).dot(centroids) / np.abs(A[:, None]).sum()
        centroid[centroid < 1e-10] = .0

        centered_vertices = vertices - centroid

        for v in centered_vertices:
            self._body.CreatePolygonFixture(
                shape=Box2D.b2PolygonShape(vertices=v.tolist()),
                density=self._density,
                friction=self._friction,
                restitution=self._restitution,
                radius=.001)

        self._fixture = self._body.fixtures

    @staticmethod
    def _shape_vertices() -> np.ndarray:
        raise NotImplementedError

    def draw(self, viewer):
        for fixture in self._fixture:
            vertices = [self._body.transform * v for v in fixture.shape.vertices]
            viewer.draw_polygon(vertices, filled=True, color=self._body_color)

    def plot(self, axes, **kwargs):
        from matplotlib.patches import Polygon
        defaults = dict(fill=True,
                        # edgecolor='#929591',
                        facecolor='#d8dcd6')
        for k in defaults:
            if k not in kwargs:
                kwargs[k] = defaults[k]

        if 'alpha' in kwargs:
            from matplotlib.colors import to_rgba
            # kwargs['edgecolor'] = to_rgba(kwargs['edgecolor'], kwargs['alpha'])
            kwargs['facecolor'] = to_rgba(kwargs['facecolor'], kwargs['alpha'])

        for fixture in self._fixture:
            vertices = [self._body.transform * v for v in fixture.shape.vertices]
            axes.add_patch(Polygon(xy=np.array(vertices), **kwargs))


class Triangle(Polygon):
    @staticmethod
    def _shape_vertices():
        return np.array([[(-0.5, 0.0),
                          (0.0, 0.0),
                          (0.0, 1.0)]])


class LForm(Polygon):
    @staticmethod
    def _shape_vertices():
        return np.array([[(-0.05, 0.0), (0.1, 0.0), (0.1, 0.3), (-0.05, 0.3)],
                         [(0.1, 0.0), (0.1, -0.15), (-0.2, -0.15), (-0.2, 0.0)]])


class TForm(Polygon):
    @staticmethod
    def _shape_vertices():
        return np.array([[(0.0, 0.15), (0.1, 0.15), (0.1, -0.15), (0.0, -0.15)],
                         [(0.0, 0.05), (0.0, -0.05), (-0.2, -0.05), (-0.2, 0.05)]])


class CForm(Polygon):
    @staticmethod
    def _shape_vertices():
        return np.array([[(0.09, 0.15), (0.09, -0.15), (-0.01, -0.15), (-0.01, 0.15,)],
                         [(-0.01, -0.15), (-0.11, -0.15), (-0.11, -0.08), (-0.01, -0.05)],
                         [(-0.01, 0.15), (-0.11, 0.15), (-0.11, 0.08), (-0.01, 0.05)]])
