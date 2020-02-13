# PROCESAMIENTO DIGITAL DE SEÑALES (2do CUATRIMESTRE 2019)
# TP2 - FERREYRA - TAPIA

import librosa as lb
import numpy as np
import matplotlib.pyplot as plt

# CARGA DE SEÑALES
(med1m,FS1m) = lb.load('med1m.wav',sr=None)
(med2m,FS2m) = lb.load('med2mruido.wav',sr=None)
(cal, FScal) = lb.load('Calibracion.wav',sr=None)

## RUIDO EN AMBAS MEDICIONES -> SE RECORTA 1ER SEGUNDO
#med1m = med1m[FS1m::]
#med2m = med2m[FS2m::]

# EMPAREJAMIENTO DE SEÑALES
n1 = len(med1m)
n2 = len(med2m)
if n1 > n2:
    med1m = med1m[0:n2]
    n = n2
elif n1 < n2:
    med2m = med2m[0:n1]
    n = n1
elif n1 == n2:
    n = n1

# DATOS DE LA SEÑAL DE ORIGEN
F0 = 100
Fr = [F0,2*F0,3*F0,4*F0,5*F0]
Ar = [.7,.4,.5,.2,.3]
t = np.linspace(0,n/FS1m,n)

### IMPORTANTE ###
# A partir del dato de la fundamental, se conoce el período mas largo.
# T0 = 1/100 = 0,01s
# Muestreado a 44100Hz ese período equivale a 441 muestras.
# Esto será dato para el cálculo de valor eficaz de las señales
n0 = int(FS1m/F0)
trms = np.linspace(0,n*n0/FS1m,n//n0)

# FUENTE MONOPOLAR EN CAMPO LIBRE
def x_distancia(r):
    x = np.zeros(n)
    for i in range(len(Fr)):
        x = x + Ar[i]*np.cos(2*np.pi*t*Fr[i])*np.exp(-1j*r*Fr[i]/343)
    return x.real

def dx_distancia(r):
    dx = np.zeros(n)
    for i in range(len(Fr)):
        dx = dx - 2*Ar[i]*Fr[i]*np.pi*np.exp(-Fr[i]*r*1j/343)*np.sin(2*np.pi*Fr[i]*t)
    return dx.real

# FUNCIONES
def rms(x,nT,n):
    x_rms = np.zeros(n//nT)
    for j in range(int(n//nT)):
        x_rms[j] = np.sqrt(sum((abs(x[j*nT:(j+1)*nT]))**2)/nT)
    return x_rms

def media(x):
    return sum(x)/len(x)

def varianza(x):
    return sum((x-media(x))**2)/(len(x)-1)

def calibracion(x):
    return x/rms(cal,len(cal),len(cal))

def dBSPL(x):
    Pref = 20*(10**(-6))
    return 20*np.log10(x/Pref)

def PtodB(x,nT,n):
    return dBSPL(calibracion(rms(x,nT,n)))

def kalman1(entrada,var_entrada,medicion,modelo1,ruido=0):
    # MODELO 1
    apriori1 = entrada + modelo1
    var_apriori1 = var_entrada + varianza(modelo1)
    K1 = var_apriori1 / (var_apriori1 + varianza(medicion) + varianza(ruido))
    aposteriori1 = apriori1 + K1*(medicion - apriori1)
    var_aposteriori1 = (1-K1)*var_apriori1
    return aposteriori1,apriori1,var_aposteriori1,K1

def kalman2(entrada,var_entrada,medicion,modelo2,ruido=0):
    # MODELO 2
    apriori2 = entrada - modelo2    
    var_apriori2 = var_entrada + varianza(modelo2)
    K2 = var_apriori2 / (var_apriori2 + varianza(medicion) + varianza(ruido))
    aposteriori2 = apriori2 + K2*(medicion - apriori2)
    var_aposteriori2 = (1-K2)*var_apriori2
    # SALIDAS
    return aposteriori2,apriori2,var_aposteriori2,K2

# PARÁMETROS PRINCIPALES  PASO A dB
m1_dB = PtodB(med1m,n0,n)
m2_dB = PtodB(med2m,n0,n)
x1_dB = PtodB(x_distancia(1),n0,n)
x2_dB = PtodB(x_distancia(2),n0,n)
dx1_dB = PtodB(dx_distancia(1),n0,n)
dx2_dB = PtodB(dx_distancia(2),n0,n)

# PARÁMETROS INICIALES DEL FILTRO (dB)
aprox = m1_dB
var = varianza(aprox)
medicion = m2_dB
x_modelo = x2_dB - x1_dB
dx_modelo = dx2_dB - dx1_dB

## PARÁMETROS INICIALES DEL FILTRO (Pa)
#aprox = med1m
#var = varianza(aprox)
#medicion = med2m
#x_modelo = x_distancia(2) - x_distancia(1)
#dx_modelo = dx_distancia(2)- dx_distancia(1)

# RUIDO ALEATORIO
ruido = np.random.normal(0,1,n)

### FILTRADO ###
veces = 100
(aprox,sp0,var,K1) = kalman1(aprox,var,medicion,x_modelo,ruido)
for ejecution in range (veces):
    (aprox,sp,var,K2) = kalman2(aprox,var,medicion,dx_modelo,ruido)

## PASO A dB (SOLAMENTE NECESARIO SI SE FILTRA EN Pa)
#aprox = PtodB(aprox,n0,n)
#sp0 = PtodB(sp0,n0,n)

# GRÁFICOS
plt.title('MEDICIÓN A 1M (SIN RUIDO)')
plt.plot(m1_dB)
plt.axis([0,n//n0,60,190])
plt.show()
plt.title('MEDICIÓN A 2M (CON RUIDO)')
plt.plot(m2_dB)
plt.axis([0,n//n0,60,190])
plt.show()
plt.title('APRIORI INICIAL')
plt.plot(sp0)
plt.axis([0,n//n0,60,190])
plt.show()
plt.title('APRIORI')
plt.plot(sp)
plt.axis([0,n//n0,60,190])
plt.show()
plt.title('APOSTERIORI')
plt.plot(aprox)
plt.axis([0,n//n0,60,190])
plt.show()