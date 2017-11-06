import numpy as np
import Box2D

from gym.envs.classic_control import rendering


class Body:
    _density = 2
    _friction = 0.01
    _restitution = 0.0

    _linear_damping = 8
    _angular_damping = 8

    def __init__(self, world: Box2D.b2World, position=None, rotation=None):
        if self.__class__ == Body:
            raise NotImplementedError('Abstract class Body cannot be instantiated.')
        self._body_color = np.array((93, 133, 195)) / 255
        self._highlight_color = np.array((238, 80, 62)) / 255

        if position is None:
            position = [.0, .0]

        if rotation is None:
            rotation = .0

        # self.scale_sim_to_vis = (1.0 / scale_real_to_sim) * scale_real_to_vis
        # self.scale_real_to_sim = scale_real_to_sim

        self._world = world
        self._body = world.CreateDynamicBody(
            position=Box2D.b2Vec2(*position),
            angle=rotation,
            linearDamping=self._linear_damping,
            angularDamping=self._angular_damping)
        self._body.linearVelocity = Box2D.b2Vec2(*[.0, .0])
        self._body.angularVelocity = .0

    def __del__(self):
        self._world.DestroyBody(self._body)

    def get_position(self):
        return np.array([self._body.position])  # / scale_real_to_sim

    def get_orientation(self):
        return self._body.angle

    def draw(self, screen):
        raise NotImplementedError('The draw method needs to be implemented by the subclass of Body.')


class Quad(Body):
    def __init__(self, width, height, **kwargs):
        super().__init__(**kwargs)

        self._width = width
        self._height = height

        self._fixture = self._body.CreatePolygonFixture(
            box=Box2D.b2Vec2(self._width/2, self._height/2),
            density=self._density,
            friction=self._friction,
            restitution=self._restitution)

    def draw(self, viewer: rendering.Viewer):
        # h = viewer.height
        # s = self.scale_sim_to_vis

        vertices = [self._body.transform * v for v in self._fixture.shape.vertices]
        # vertices = [(s * x, h - s * y) for (x, y) in vertices]

        viewer.draw_polygon(vertices, filled=True, color=self._body_color)


class CornerQuad(Quad):
    def draw(self, viewer: rendering.Viewer):
        super(CornerQuad, self).draw(viewer)

        # h = viewer.height
        # s = self.scale_sim_to_vis

        vertices = [self._body.transform * v for v in self._fixture.shape.vertices]
        # vertices = [(s * x, h - s * y) for (x, y) in vertices]

        viewer.draw_polygon(vertices[0:3], filled=True, color=self._highlight_color)


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

    def draw(self, viewer: rendering.Viewer):
        t = rendering.Transform(translation=self._body.position)
        viewer.draw_circle(self._radius, res=100, color=self._body_color).add_attr(t)


class LetterForm(Body):
    def __init__(self, width, height, **kwargs):
        super().__init__(**kwargs)

        self._width = width
        self._height = height
        self._fixture = []

    def draw(self, viewer: rendering.Viewer):
        for fixture in self._fixture:
            vertices = [self._body.transform * v for v in fixture.shape.vertices]
            # vertices = [(s * x, h - s * y) for (x, y) in vertices]

            viewer.draw_polygon(vertices, color=self._body_color)
            t = rendering.Transform(translation=self._body.position)
            viewer.draw_circle(10, 30, color=np.array((0, 255, 127))/255).add_attr(t)

            # for a nice anti-aliased object outline
            # gfxdraw.aapolygon(screen, verts, self.object_color)
            # gfxdraw.filled_polygon(screen, verts, self.object_color)
            # gfxdraw.circle(screen, int(self.body.position[0] * s), int(h - self.body.position[1] * s), 10,
            #                (0, 255, 127, 255))


class LForm(LetterForm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        v1 = np.array([(-0.075, 0), (-0.075, -0.1), (0.125, -0.1), (0.125, 0)])
        v1 *= [self._width / 2, self._height / 3]
        v2 = np.array([(-0.075, 0), (-0.075, 0.2), (0.025, 0.2), (0.025, 0)])
        v2 *= [self._width / 2, self._height / 3]

        self._body.CreatePolygonFixture(
            shape=Box2D.b2PolygonShape(vertices=v1.tolist()),
            density=self._density,
            friction=self._friction,
            restitution=self._restitution)
        self._body.CreatePolygonFixture(
            shape=Box2D.b2PolygonShape(vertices=v2.tolist()),
            density=self._density,
            friction=self._friction,
            restitution=self._restitution)

        self._fixture = self._body.fixtures


class TForm(LetterForm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        v1 = np.array([(0.15, 0.025), (-0.15, 0.025), (-0.15, -0.075), (0.15, -0.075)])
        v1 *= [self._width / 3, self._height / 2]
        v2 = np.array([(0.05, 0.125), (0.05, 0.025), (-0.05, 0.025), (-0.05, 0.125)])
        v2 *= [self._width / 3, self._height / 2]

        self._body.CreatePolygonFixture(
            shape=Box2D.b2PolygonShape(vertices=v1.tolist()),
            density=self._density,
            friction=self._friction,
            restitution=self._restitution)
        self._body.CreatePolygonFixture(
            shape=Box2D.b2PolygonShape(vertices=v2.tolist()),
            density=self._density,
            friction=self._friction,
            restitution=self._restitution)

        self._fixture = self._body.fixtures


class CForm(LetterForm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        v1 = np.array([(0.15, 0.01), (-0.15, 0.01), (-0.15, -0.09), (0.15, -0.09)])
        v1 *= [self._width / 3, self._height / 2]
        v2 = np.array([(-0.15, 0.01), (-0.15, 0.11), (-0.08, 0.11), (-0.05, 0.01)])
        v2 *= [self._width / 3, self._height / 2]
        v3 = np.array([(0.15, 0.01), (0.15, 0.11), (0.08, 0.11), (0.05, 0.01)])
        v3 *= [self._width / 3, self._height / 2]

        self._body.CreatePolygonFixture(
            shape=Box2D.b2PolygonShape(vertices=v1.tolist()),
            density=self._density,
            friction=self._friction,
            restitution=self._restitution)
        self._body.CreatePolygonFixture(
            shape=Box2D.b2PolygonShape(vertices=v2.tolist()),
            density=self._density,
            friction=self._friction,
            restitution=self._restitution)
        self._body.CreatePolygonFixture(
            shape=Box2D.b2PolygonShape(vertices=v3.tolist()),
            density=self._density,
            friction=self._friction,
            restitution=self._restitution)

        self._fixture = self._body.fixtures
