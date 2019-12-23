from flask import Flask
from GenAbstract import GenAbstract

app = Flask(__name__)
logger = app.logger

genAbstract = GenAbstract(model_name='/Users/ever/Documents/AI/NLP课程/projects/1/GenAbstract/result/wiki_1219.model')

from app import routes
