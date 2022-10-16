import taichi as ti
N = int(input("enter N:"))
R = max(4,N//100)
BOUND = False
SPHERE = True
radius = 0.3
dt = 0.005
G = 1
cam_speed = 0.1
paused = True

ti.init(ti.cuda)
pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
vel = ti.Vector.field(3, dtype=ti.f32, shape=N)
force = ti.Vector.field(3, dtype=ti.f32, shape=N)
mass = ti.field(ti.f32, shape=N)
# radius = ti.field(ti.f32, shape=N)

@ti.kernel
def compute_force(GC: ti.f32):
    for i in range(N):
        force[i] = ti.Vector([.0,.0,.0])
    for i in range(N):
        for j in range(N):
            if i != j:
                r = pos[j] - pos[i]
                force[i] += GC*mass[i]*mass[j]*r/(r.norm(1e-2)**3)

@ti.kernel
def update(dt: ti.f32):
    for i in range(N):
        vel[i] += force[i]*dt/mass[i]
        pos[i] += vel[i]*dt
        if BOUND:
            for k in ti.static(range(3)):
                if pos[i][k]<-R:
                    pos[i][k] = (pos[i][k]+R)*(-1)-R
                    vel[i][k]*=-1
                if pos[i][k]>R:
                    pos[i][k] = 2*R-pos[i][k]
                    vel[i][k]*=-1

@ti.kernel
def initialize():
    for i in range(N):
        tpos = ti.Vector([ti.random()*2-1, ti.random()*2-1, ti.random()*2-1])
        if SPHERE:
            tpos/=tpos.norm()
        tpos *= R
        pos[i] = tpos
        mass[i] = 30

def show_options():
    global dt
    global G
    global radius
    global paused
    global cam_speed
    window.GUI.begin("Options", 0.05, 0.3, 0.25, 0.2)
    G = window.GUI.slider_float("G", G, 0, 10)
    radius = window.GUI.slider_float("Radius", radius, 0, 3)
    dt = window.GUI.slider_float("dt", dt, 0, 0.02)
    cam_speed = window.GUI.slider_float("vCamera", cam_speed, 0, 3)
    if window.GUI.button("Reset"):
        initialize()
        for i in range(N):
            force[i] = ti.Vector([.0, .0, .0])
            vel[i] = ti.Vector([.0, .0, .0])
    if paused:
        if window.GUI.button("Continue"):
            paused = False
    else:
        if window.GUI.button("Pause"):
            paused = True
    window.GUI.end()

def render():
    camera.track_user_inputs(window, movement_speed=cam_speed, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.ambient_light((0.8, 0.8, 0.8))
    scene.particles(pos, radius=radius, color=(.1,0.8,0.8))

    scene.point_light(pos=(R, R, R), color=(1, 1, 1))
    scene.point_light(pos=(-R, -R, -R), color=(1, 0, 0))
    canvas.scene(scene)

initialize()

window = ti.ui.Window("N body simulation!", (1080, 720))
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(R*3, 0.0, 0.0)
camera.lookat(0.0, 0.0, 0.0)
camera.fov(60)
canvas = window.get_canvas()


while window.running:
    if window.get_event(ti.ui.PRESS):
        if window.event.key == 'p':
            paused = not paused
            print(paused)
        elif window.event.key == 'r':
            initialize()
            for i in range(N):
                force[i] = ti.Vector([.0,.0,.0])
                vel[i] = ti.Vector([.0,.0,.0])

    if not paused:
        compute_force(G)
        update(dt)
    render()
    show_options()
    window.show()