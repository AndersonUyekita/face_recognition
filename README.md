# Face Recognition

Esse _script_ visa utilizar câmeras de vigilância para identificar as pessoas.

## 1. Requisitos

Vamos utilizar _packages_ disponíveis na comunidade Python. Desta forma, a biblioteca `face_recognition` será implementada.

### 1.1. Preparativos

Será necessário a prévia instalação do `CMAKE`, a forma mais fácil encontrada foi instando o `Visual Studio Build Tools` ou atualizando o `Visual Studio`.

Após a instalação/atualização do `Visual Studio` será possível instalar o `CMAKE` no ambiente Python pelo `prompt` do Anaconda.

```
pip install cmake
```

O `CMAKE` é usado para "compilar" o _package_ `dlib` que possui a particularidade de parte de seu código ser em `C++`.

```
conda install -c conda-forge dlib
```

Os demais _packages_ são de simples instalação.

### 1.2. Bibliotecas

Haverá apenas a necessidade da instalação dos _packages_:

* [`cv2`](https://pypi.org/project/opencv-python/): Biblioteca de visão computacional _open source_;
* [`face_recognition`](https://pypi.org/project/face-recognition/): Também é uma biblioteca _open source_ que faz todo o "trabalho" de reconhecer as faces, e;
* [`numpy`](https://pypi.org/project/numpy/): Usado para manipular dados.
* [`dlib`](https://pypi.org/project/dlib/): É um biblioteca feita em C++, é complicado de instalar no computador. Daí o motivo da seção 1.1. Preparativos.

### 1.3. _Environment_

Talvez seja necessário a criação de um ambiente específico para o projeto.

Quando tentei usar o ambiente `(base)` o _script_ alegava que não um problema de kernel.

Só consegui treinar o algoritmo após a criação do ambiente dedicado para o treinamento. Pode ser que um ambiente muito carregado como o `(base)` interfira negativamente como a instalação de _packages_ corrompidos.

## 2. Funcionamento

Diferentemente do _script_ [`ppe_surveillance`](https://github.com/AndersonUyekita/ppe_surveillance), o `face_recognition` já vem treinado e só há a necessidade de inserir uma imagem da pessoa que você quer que o _script_ reconheça.

## 3. Organização

O script ficará na raiz do repositório, sendo que as imagens serão inseridas na pasta `01-data`.
