# Netoga poravnava slik

## B-zlepki
Uporabljajo se na 2D in 3D slikah. 
Problem metode je ogromno število parametrov
Poravnavo bomo vrednostili z mero podobnosti, ki nam omogoča tudi primerjavo različnih modalitet.
Nemoreš celotno sliko določiti s točkami, saj narašča število parametrov s tem, kar rata računsko zahtevno
hkrati pa zaradi deformacije nočemo veliko število točk, saj bodo lokalne deformacije večje
Bazne funkcije B-zlepkov so odvedljive, njigova vsota pa znaša 1
B-zlepke računamo na mreži kontrolnih točk, kjer za vsako točko (i,j) računamo B-zlepek (Računamo kje se neka točka nahaja)

Ko imamo točke določimo kam se preslikajo nato izvedemo interpolacijo