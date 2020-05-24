A backend modulnak innen töltöttem le a:

https://drive.google.com/drive/folders/1pQNZ9snByUOMjvEf7Td8Zg1qvBAVhWZ8

Az rbc.h5-öt

A generált fileok 20 százalékát áttettem validation mappákba, de mivel 100 filenál nem enged meg többet feltölteni egyszerre a github, egyelőre ezeket még nem töltöttem fel.

A config_rbc.json-t futtatom.

# A hibaüzenet:

Fused conv implementation does not support grouped convolutions for now.

Elvileg meg kéne változtatnom a placeholder, amit nem tudok, hogy hol van.

Azt is találtam, hogy talán valamiért nem fut a GPU-n a program, de ha CPU-n futtatom, akkor is kidobja ezt a hibaüzenetet.

A CUDA-t, illetve a cudnn-t felraktam a számítógépemre.

Az általam használt GPU egy Nvidia GeForce GTX 1660


# Megjegyzés:
A legújabb tensorflowal nem működik, mert miért lenne visszafelé kompatibilis?
