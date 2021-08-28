from logging import debug
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model
#Para utilizar este import es necesario instalar PIL.
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

#Para utilizar el método secure_files de Flask con el que subir imágenes a la web, es necesario 
#añadir una clave secreta.

app.config.update(SECRET_KEY=os.urandom(24))

#Redes desarrolladas en la experimentación.
redes = ['primer_red.h5','segunda_red.h5','vgg_red.h5','inception_red.h5','resnet_red.h5']

#Carpeta donde se guardarán las imágenes.
UPLOAD_FOLDER = str(os.getcwd() + '/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#Formatos de imágenes aceptadas.
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

#Comprueba que la imagen posee una de las extensiones permitidas
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
#Vista inicial de la página, sin subir el formulario. Se envía la variable redes para 
#poder seleccionar en el formulario las redes.
@app.route('/')
def upload_form():
	return render_template('form.html', redes = redes)

#Vista tras enviar el formulario.
@app.route('/', methods=['POST'])
def upload_image():
	#Si no se rellena el formulario te devuelve a la vista inicial.
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	#Nombre de la imagen.
	file = request.files['file']
	#Red seleccionada en la vista inicial.
	red = request.form["red"]
	#Umbral de decisión seleccionado en la vista inicial.
	umbral = int(request.form["umbral"])
	#Si no se manda la imagen en el formulario te redirige a la vista inicial.
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	#Si la imagen existe y es una imagen con extensión válida:
	if file and allowed_file(file.filename):
		#Mediante este método obtenemos el nombre original de la imagen. Si no lo utilizáramos, se vería reflejada
		#en la consola la ruta completa de la imagen.
		filename = secure_filename(file.filename)
		#La guardamos en la carpeta static.
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		print('imagen a tratar: ' + filename)
		print(red)

		#Cargamos el modelo seleccionado en el formulario.
		modelo = load_model(red)

		#Cargamos la imagen y la normalizamos dividiéndola entre 255, para poder alimentarla a la red.
		img = image.load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename), target_size=(150, 150))
		img_array = image.img_to_array(img)
		img_array /= 255.
		#Como el método predict de las redes realiza predicciones sobre arrays de imágenes, añadimos la única imagen
		#que tenemos a un array para poder utilizar este método.
		img_batch = np.expand_dims(img_array, axis=0)

		#Realizamos la predicción sobre la imagen.
		pred = modelo.predict([img_batch])
		
		#Cadena donde guardaremos la respuesta y probabilidad inicial.
		res=""
		prob = 0.

		#Operación ternaria, donde pasamos el número [0 - 100] del umbral en el formulario a un número [0. - 1.].
		#Si el valor recibido en el formulario es 0 lo dejamos como está, para evitar realizar una operación n/0
		um = umbral/100 if umbral != 0 else umbral

		#Si la imagen ha obtenido una predicción en la que el valor de la clase enfermo sea mayor que el umbral de decisión
		#damos por hecho que la imagen posee la clase enfermo. En otro caso es de clase sano.
		if pred[0][0]>=um:
			res = "Enfermo"
			prob = pred[0][0]
		else:
			res = "Sano"
			prob = 1. - pred[0][0]
		
		#Redondeamos la probabilidad a 7 décimas
		p = round(prob,7)

		#Devolvemos por consola que se ha realizado correctamente el proceso y devolvemos la vista con
		#el resultado del proceso.
		flash('Image successfully uploaded and displayed below')
		return render_template('form.html', filename=filename, redes=redes, res=res, red=red, p=p)
	else:
		#En otro caso, no se ha subido una imagen con una extensión permitida.
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

""" @app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301) """

if __name__ == "__main__":
    app.run(debug=True)