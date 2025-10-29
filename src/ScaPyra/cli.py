import typer
import sys

app = typer.Typer(help="CLI for controlling the SCARA robot using PCA9685 driver.")

@app.command()
def status():
    """Show current TCP position and servo state."""
    from ScaPyra.scara import SCARAController
    scara = SCARAController()
    typer.echo(f"Current position: X={scara.x:.2f}, Y={scara.y:.2f}")

@app.command()
def move(x: float, y: float):
    """Move robot TCP to absolute position (x, y)."""
    from ScaPyra.scara import SCARAController
    scara = SCARAController()
    result = scara.flat_move(x, y)
    if result:
        typer.echo(f"Moved to ({x:.2f}, {y:.2f}) successfully.")
    else:
        typer.echo(f"Failed to move to ({x:.2f}, {y:.2f}).")
        sys.exit(1)

@app.command()
def interpolate(x: float, y: float, steps: int = 10, delay: float = 0.1):
    """Move robot with linear interpolation between current and target (x, y)."""
    from ScaPyra.scara import SCARAController
    scara = SCARAController()
    ok = scara.interpolated_flat_move(x, y, steps=steps, delay=delay)
    if ok:
        typer.echo(f"Interpolated move to ({x:.2f}, {y:.2f}) completed.")
    else:
        typer.echo(f"Interpolation failed before reaching ({x:.2f}, {y:.2f}).")
        sys.exit(1)

@app.command()
def lift(time: float = 5.0, pulse: int = 1500):
    """Lift the robot vertically."""
    from ScaPyra.scara import SCARAController
    scara = SCARAController()
    ok = scara.lift_robot(lift_time=time, lift_pulse=pulse)
    if ok:
        typer.echo(f"Robot lifted (pulse={pulse}, time={time}s).")
    else:
        typer.echo("Lift failed.")
        sys.exit(1)

@app.command()
def lower(time: float = 5.0, pulse: int = 1610):
    """Lower the robot vertically."""
    from ScaPyra.scara import SCARAController
    scara = SCARAController()
    ok = scara.lower_robot(lower_time=time, lower_pulse=pulse)
    if ok:
        typer.echo(f"Robot lowered (pulse={pulse}, time={time}s).")
    else:
        typer.echo("Lowering failed.")
        sys.exit(1)

@app.command()
def set_angle(angle: int, motor: str):
    """Set angle on chosen motor"""
    from ScaPyra.scara import SCARAController
    scara = SCARAController()
    pulse = scara.angle_to_pulse(angle=angle, motor=motor)
    scara.pwm.setServoPulse(scara.motor1_channel, pulse)
    typer.echo("Done")

if __name__ == "__main__":
    app()
