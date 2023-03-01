from deployment import app, db
from flask.cli import FlaskGroup
from flask_apscheduler import APScheduler
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

cli = FlaskGroup(app)

@cli.command("create_db")
def create_db():
    db.drop_all()
    db.create_all()
    db.session.commit()


scheduler = APScheduler()
scheduler.init_app(app)
#scheduler.add_job(id="model", func=fetch_model, trigger='interval', days=1)
scheduler.start()

if __name__ == '__main__':
    cli()