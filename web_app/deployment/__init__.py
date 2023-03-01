from flask_sqlalchemy import SQLAlchemy
from flask import Flask
import os

app = Flask(__name__)
app.config.from_object("deployment.config.Config")
db = SQLAlchemy(app)


class Detections(db.Model):
    __tablename__ = "detections"

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(128), unique=True, nullable=False)
    longitude = db.Column(db.Numeric(precision=8))
    latitude = db.Column(db.Numeric(precision=8))


from deployment import routes