# Classification model

## 1. Priprava podatkov 

3 množice podatkov:
- Učna 
- Testna (Drop out množica)
- Validacijska

Učna in validacijska zajemata 66% ostalo testna. 44% učna in 22% validacijska.

Dimenzija slike:
(število pacientov, x, y, ali je slika t1 ali flair)
Lahko se odločimo da uporabimo samo t1, samo flair ali oboje

Normalizacija intenzitet slik:
Slike normaliziramo, zaradi stabilnosti (da niso velike razlike med vrednostmi)
Ena opcija je da vsakemu voxlu odštejemo srednjo vrednost in delimo s standardno diviacijo.

## 2. Razvrščanje podatkov (Klasifikacija)

Pove nam v kateri razred pade dan podatek.

Prvi korak je da inicializiraš uteži. Za začetne lahko uporabiš random seed. Boljša kot bo začetna inicializacija, boljše bo učenje -> Good luck da nastaviš

## 3. Razgradnja slik

### Unet

## 4. Docker
Docker image se zgradi po recepturi *Dockerfile*. Primer:

```dockerfile
# Osnovna slika iz Docker Hub-a
FROM ubuntu:20.04

# Knjižnice
RUN apt-get update && apt-get install -y numpy scikit-learn matplotlib

# Delovna mapa in naša koda
WORKDIR /app
COPY ./myapp /app

# Zagonski ukaz
CMD ["./start-myapp.sh"]
```

To build a image run:

```
docker build -t <ImageName> -f <Dockerfile_name> .
```
If you have one Dockerfile you dont need to specify -f <Dockerfile_name>

To run the image use:

```
docker run -i -t --rm image_name python main.py
docker run --rm --runtime=nvidia image_name python main.py
```
-i -t pomeni interaktivni ukazni način
--rm pomeni, da bo odstranjen ko končamo

Attach to VScode, da se povežeš v container. Tu lahko izzvajaš debug and testing

Mapiranje
```
docker run .i .t --rm -d -v ./moja_mapa:/workdir/source/moja_mapa my_python_image bash
```