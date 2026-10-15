PY ?= python
P  ?= 4

.PHONY: gui gui-mpi pruebas demo medir medir-rapido graficas capturas informe todo limpiar

gui:            ## Interfaz gráfica (secuencial / hilos / nativo)
	$(PY) -m vida.gui --tamano 500

gui-mpi:        ## Interfaz gráfica de la versión MPI con P procesos (make gui-mpi P=4)
	mpiexec -n $(P) $(PY) -m vida.gui_mpi --tamano 500

pruebas:        ## Pruebas de correctitud
	$(PY) -m pytest -q

demo:           ## Demostración de condición de carrera (sin barrera)
	$(PY) bench/demo_condicion_carrera.py

medir:          ## Todas las mediciones + gráficas (varios minutos)
	$(PY) bench/ejecutar_todo.py

medir-rapido:   ## Mediciones cortas para probar que todo funciona
	$(PY) bench/ejecutar_todo.py --rapido

graficas:
	$(PY) bench/graficas.py

capturas:       ## Capturas de la GUI para el informe
	$(PY) bench/capturas.py

informe:        ## Compila informe/informe.pdf a partir de resultados/
	$(PY) informe/generar_datos.py
	cd informe && latexmk -pdf -interaction=nonstopmode informe.tex

todo: pruebas medir capturas informe

limpiar:
	cd informe && latexmk -C
	rm -rf **/__pycache__ .pytest_cache
