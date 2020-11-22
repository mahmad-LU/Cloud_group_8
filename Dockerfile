FROM python:3.8


# Add the application source code.
ADD . /rundir
#Name the workdir that we need.
ADD requirements.txt /rundir/requirements.txt
RUN pip install -r /rundir/requirements.txt

WORKDIR /rundir

EXPOSE 5000

# ENTRYPOINT FLASK_APP=app.py flask run --host=0.0.0.0
CMD flask run -h 0.0.0.0 -p 5000