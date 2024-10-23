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
