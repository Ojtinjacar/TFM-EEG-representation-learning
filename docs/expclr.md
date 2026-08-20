# ExpCLR: el algoritmo y su implementación en este repositorio

Documenta la réplica de **ExpCLR** (Nonnenmacher et al., *Utilizing Expert Features for Contrastive
Learning of Time-Series Representations*, ICML 2022) aplicada a EEG infantil para estimar madurez
neurocognitiva. Las referencias `(p. N)` apuntan a las páginas del PDF en
`refs/papers/nonnenmacher2022expclr_arxiv.pdf`.

Índice: [1. Qué resuelve](#1-qué-problema-resuelve-expclr) · [2. El algoritmo](#2-el-algoritmo)
· [3. Qué promete y qué no se puede medir](#3-qué-promete-el-método-y-qué-no-podemos-medir)
· [4. Por qué Delta = 1 degenera](#4-por-qué-delta--1-degenera-con-nuestro-descriptor)
· [5. Mapa del código](#5-mapa-del-código) · [6. Desviaciones](#6-desviaciones-declaradas-respecto-al-paper)
· [7. Cómo ejecutarlo](#7-cómo-ejecutarlo)

---

## 1. Qué problema resuelve ExpCLR

El aprendizaje contrastivo clásico (SimCLR y derivados) necesita **augmentations**: transformar una
muestra de dos formas distintas y pedir al encoder que acerque las dos vistas. Eso exige saber de
antemano qué transformaciones preservan la clase, algo que en series temporales biomédicas no es
obvio. Aplicar ruido gaussiano o desplazamientos temporales a un EEG infantil puede destruir
justamente la información madurativa que se busca.

ExpCLR sustituye ese conocimiento experto implícito por uno **explícito**: un vector de
características diseñadas por especialistas del dominio. En lugar de "estas dos vistas son la misma
muestra, acércalas", el objetivo pasa a ser "estas dos muestras tienen descriptores expertos
parecidos, así que sus representaciones deben estar igual de cerca".

> "Our method employs expert features to **replace the commonly used data transformations** in
> previous contrastive learning approaches." (p. 1)

Consecuencias prácticas: **una sola vista por muestra y cero augmentations**. El contraste ocurre
entre muestras distintas del mismo lote, ponderado por lo parecidos que sean sus descriptores.

En este proyecto el descriptor es `P_diverso`, 40 columnas repartidas en 5 por cada una de 8
familias del catálogo de features v2, elegidas **sin mirar la edad** para que la geometría no
colapse en una sola dirección.

---

## 2. El algoritmo

Notación: un lote de `N` ventanas. `x_i` es la ventana, `f_i` su descriptor experto de dimensión
`d`, y `E(x_i)` su representación de dimensión `e`.

### 2.1 Similitud entre descriptores

Se mide cuánto se parecen dos ventanas **según el experto**, normalizando por la distancia máxima
del conjunto para que caiga en `[0, 1]`:

```
s_ij = ( 1 - ||f_i - f_j|| / max_kl ||f_k - f_l|| )^2        (Eq. 5, p. 5)
```

El cuadrado envuelve **todo el paréntesis**, no solo la norma. La versión sin cuadrado existe
(Eq. 2) y el paper la evalúa como ablación, pero rinde peor: *"due to its consistent superior
performance in experiments"* (p. 5). En el código, `--linear_similarity` selecciona la variante
lineal.

`s_ij = 1` significa descriptores idénticos, `s_ij = 0` los dos más lejanos del conjunto. El
denominador puede calcularse sobre todo el conjunto de entrenamiento o por lote; el paper permite
ambas (p. 4) y aquí es la opción `--sim_max`.

### 2.2 Distancia normalizada entre representaciones

```
mu_i = (1/N) * sum_j ||E(x_i) - E(x_j)||
D_ij = ||E(x_i) - E(x_j)|| / mu_i                            (Sec. 3.6, p. 5)
```

`mu_i` es la distancia media **desde el punto i a todos los demás**, incluido él mismo (que aporta
cero). Dos consecuencias que conviene tener presentes porque explican casi todo lo que sigue:

- **`D_ij` es asimétrica**: `D_ij != D_ji`, porque cada fila se normaliza por su propia media.
- **Cada fila de `D` suma exactamente `N`**, por construcción. No es una tendencia, es una
  identidad algebraica: la media de la fila `i` vale 1 siempre.

### 2.3 Término por par y pérdida

Para cada par se mide cuánto se desvía la distancia observada de la deseada:

```
L_ij = ( (1 - s_ij) * Delta - D_ij )^2
```

La diana es `(1 - s_ij) * Delta`: si los descriptores son idénticos (`s_ij = 1`) la distancia
deseada es cero, y si son lo más distintos posible (`s_ij = 0`) es `Delta`. La pérdida final agrega
esos términos con un log-sum-exp:

```
L = tau * log( sum_ij exp(L_ij / tau) / N^2 )                (Eq. 4, p. 5)
```

El sumatorio recorre **todos los pares, incluida la diagonal** `i = j` (donde `L_ii = 0`), y divide
entre `N^2`.

### 2.4 Por qué el log-sum-exp: minado implícito de negativos duros

El log-sum-exp es una media suavizada que pondera más los términos grandes. La Proposición 3 (p. 5)
da los dos extremos:

- **`tau -> 0`**: equivale a minimizar `max_ij L_ij`, solo el peor par.
- **`tau -> infinito`**: equivale a la media, es decir la pérdida cuadrática simple (Eq. 3).

Con `tau = 1` se obtiene un punto intermedio: los pares peor ajustados reciben más gradiente sin que
los demás desaparezcan. El apéndice F lo hace explícito: el gradiente respecto a `L_nm` es
proporcional a `exp(L_nm / tau)`, un softmax sobre los términos por par. Eso es minado de negativos
duros sin tener que seleccionarlos a mano.

La bandera `--no_hard_negative_mining` activa la Eq. 3 pura (la ablación NHNM del paper).

---

## 3. Qué promete el método, y qué no podemos medir

El paper justifica ExpCLR con dos propiedades. Una representación es **bilipschitz** con constantes
`l_-` y `l_+` si preserva las distancias del espacio de descriptores dentro de esos factores. La
Proposición 2 (p. 5) dice que si la pérdida cuadrática llega a cero, entonces `E` es bilipschitz con
`l_- = l_+ = max_ij ||f_i - f_j|| / Delta`.

**Aquí esa garantía no es invocable, por dos razones independientes.**

**Primera: la pérdida es invariante al reescalado del embedding.** Si `E -> cE`, entonces las
distancias se multiplican por `c`, pero `mu_i` también, así que `D_ij` no cambia y la pérdida
tampoco. Está verificado en `tests/test_expclr_loss.py::test_loss_is_invariant_to_rescaling_the_embedding`:
la pérdida coincide para `c` entre 1e-3 y 1e3. Como cualquier escala da el mismo valor, las
constantes `l_-` y `l_+` quedan **indeterminadas**. La pérdida restringe la *forma* de la nube de
puntos, nunca su tamaño.

La laguna está localizada, y **es del paper, no de la réplica**. La demostración de la Proposición 2
(App. C.2, p. 18) parte de la Eq. 2, la similitud lineal, y de `D_ij` **sin normalizar**: de ahí
deduce `(Delta / max_kl ||f_k - f_l||) * ||f_i - f_j|| = ||E_i - E_j||`. Después el paper añade que
*"it is easy to see that other similarity measures like the one introduced in Sec. 3.6 also provide
good bounds"*, pero eso cubre la **medida de similitud**, no la **normalización por `mu_i`**, que es
lo que introduce la invariancia. Nunca se demuestra que la garantía sobreviva a esa normalización.

Lo mismo ocurre con el apéndice D, que define la constante pair-Lipschitz
`Z_ij = ||f_i - f_j|| / ||E(x_i) - E(x_j)||` también **sin normalizar**. Bajo `mu_i` esa cantidad no
está determinada, así que las cotas PAC de las ecuaciones 7 y 8 no son calculables aquí.

Por eso el diagnóstico de este repositorio usa **dimensionalidad efectiva**, que es adimensional, en
lugar de las cotas del apéndice D.

**Segunda: la hipótesis de la proposición es inalcanzable con `Delta = 1`.** Es el asunto de la
sección siguiente.

---

## 4. Por qué `Delta = 1` degenera con nuestro descriptor

Este es el hallazgo central de la réplica, y la razón de que la primera ejecución diera un encoder
preentrenado **peor que uno con pesos aleatorios**.

### El argumento

Por la sección 2.2, cada fila de `D` suma exactamente `N`. Por otro lado, la diana de esa misma fila
suma como mucho `Delta * (N - 1)`, porque el término diagonal se anula (`s_ii = 1`) y cada sumando
restante es a lo sumo `Delta`.

Con `Delta = 1` la diana suma como mucho `N - 1`, mientras la distancia real suma `N`. **La pérdida
no puede llegar a cero**: haría falta `Delta >= N/(N-1)`. Verificado en
`test_zero_loss_is_unreachable_at_delta_one`.

Y no se trata solo de que el mínimo no sea cero. Midiendo con `P_diverso`:

| Magnitud | Valor |
|---|---|
| Media de `(1 - s_ij)` con `--sim_max train` | 0.537 |
| Desajuste por par con `Delta = 1` | 0.463 |
| Fracción irreducible de la pérdida | 93 % a 95 % |
| Suelo alcanzable (geometría equidistante) | 0.2228 en forma Eq. 3, 0.2238 en Eq. 4 |
| Pérdidas realmente observadas | 0.2615, 0.2392, 0.2337 |

La descomposición es una identidad de Pitágoras exacta. Escribiendo `t_i` como la media de la fila
`i` de la diana, y usando que la media de fila de `D` es 1:

```
media_j (T_ij - D_ij)^2 = (t_i - 1)^2 + media_j[ (T_ij - t_i) - (D_ij - 1) ]^2
                          ^ sesgo        ^ forma de la geometría
```

El término de sesgo **no contiene a `D`**: ningún encoder puede reducirlo. Medido, vale 0.2194 y es
idéntico para el encoder entrenado, el aleatorio y la proyección.

### El efecto observado

El óptimo alcanzable con `Delta = 1` es la geometría **equidistante**, con todos los pares a la
misma distancia, que es la menos informativa posible. Y el encoder llega ahí:

| Época | Pérdida | Dimensionalidad efectiva | R2 en holdout |
|---|---|---|---|
| 0 (aleatorio) | — | 8.76 | 0.401 |
| 1 | 0.7301 | 2.12 | 0.021 |
| 12 | 0.2862 | 3.03 | 0.151 |

La primera época contrae la nube de 8.76 a 2.12 direcciones efectivas. **La pérdida bajaba mientras
la representación se destruía**, que es exactamente por qué el entrenamiento registra ahora la
dimensionalidad efectiva junto a la pérdida.

### El ajuste, y por qué `Delta = 1` funciona en el paper

Igualar las medias de fila da `Delta* = 1 / E[1 - s_ij]`, que con este descriptor vale entre 1.84 y
1.95 según cómo se estime. Con `Delta = 1.86` medido sobre tres épocas: el suelo cae de 0.2154 a
0.0251 y la dimensionalidad efectiva **sube** de 8.51 a 26.31 en lugar de caer.

La pregunta obvia es por qué a los autores les sirve `Delta = 1`. La respuesta está en la
**concentración de distancias**, y explica el mecanismo completo. Midiendo `Delta*` sobre
descriptores gaussianos de la dimensión de cada conjunto del paper:

| Dimensión del descriptor | `d_media / d_max` | `E[1 - s_ij]` | `Delta*` |
|---|---|---|---|
| 29 (SleepEDF) | 0.602 | 0.821 | 1.22 |
| 176 (Waveform) | 0.785 | 0.936 | 1.07 |
| 561 (HAR) | 0.866 | 0.966 | 1.04 |
| **`P_diverso` real (40)** | **0.325** | **0.532** | **1.88** |

En dimensión alta las distancias se concentran, el cociente `d_media / d_max` tiende a uno, la
similitud tiende a cero y **`Delta*` tiende a uno por sí solo**. `Delta = 1` no es una constante
arbitraria del método: es donde caen los descriptores del paper.

`P_diverso` se comporta distinto **no por tener 40 columnas sino por estar correlacionado**: su
dimensionalidad efectiva es 21.51 de 40, frente a 39.70 de un gaussiano independiente de la misma
forma. Sus distancias no se concentran, la similitud media sube y `Delta*` se va a 1.88.

**Y 1.88 cae dentro del rango que el propio paper valida.** El panel izquierdo de la figura 4
(App. A.2, p. 13) barre `Delta` en `{0.1, 0.5, 1.0, 2.0, 5.0}` y muestra una meseta entre 0.5 y 2.0
en los tres conjuntos, hundiéndose fuera de ella. El texto lo dice: *"There is an optimal value for
Delta which is robust against modifications within a certain range."*

**Qué se declara entonces.** No que el paper deje `Delta` abierto, porque no lo hace: lo fija a 1
para todos sus conjuntos *y para todos los algoritmos comparados*, con un motivo de equidad
competitiva explícito (*"to avoid overfitting ExpCLR w.r.t our competing methods"*, App. B.3). Lo
que se declara es que **`Delta` debe escalar con la concentración de distancias del descriptor**, y
que el valor usado aquí está dentro de la meseta que el paper valida. La desviación es de
procedimiento, no de rango: ellos eligen un valor común entre algoritmos, aquí se elige el que
corresponde a este descriptor.

---

## 5. Mapa del código

```
src/loss.py            ExpCLRLoss: las ecuaciones 2, 3, 4 y 5
src/train_expclr.py    preentrenamiento de un encoder + diagnósticos de geometría
src/eval_expclr.py     evaluación: extracción congelada, probes, métricas, bootstrap
tune_expclr.py         selección de Delta, lr y sim_max en un split reservado
run_e3_loso.py         orquestador: LOSO sobre todos los métodos con el mismo protocolo
```

### `src/loss.py`

`ExpCLRLoss` con tres métodos públicos: `expert_similarity` (Eq. 2/5), `normalized_distance`
(la de la Sec. 3.6) y `reduce`, que aplica la Eq. 3 o la Eq. 4 según haya temperatura. `reduce`
está expuesto para poder puntuar geometrías de referencia con **el mismo funcional** que se
optimiza; compararlas con una media cuando se entrena un log-sum-exp no dice nada, porque por
Jensen el segundo siempre es mayor.

### `src/train_expclr.py`

Preentrena un encoder. Además del bucle:

- `prepare_descriptor`: imputa por mediana y estandariza el descriptor **solo con las filas
  recibidas**, que bajo LOSO son las de entrenamiento. Sin esto, columnas cuya desviación va de
  4.1e-12 a 28.5 harían que dos o tres dominaran toda la geometría.
- `effective_dimensionality`: `exp(H)` sobre los autovalores de la covarianza. Una nube repartida
  por igual entre `k` direcciones ortogonales puntúa `k`. Es el detector de contracción.
- `equidistant_reference`: la matriz de distancias de una nube equidistante, que **no** es una
  matriz de unos sino de `N/(N-1)`, por lo dicho en la sección 2.2.
- `diagnose_epoch`: mide ambas cosas sobre **lotes fijos**, extraídos de un generador propio. Si
  iterase el DataLoader de entrenamiento consumiría su permutación y el intervalo de diagnóstico se
  convertiría en un hiperparámetro encubierto.
- `checkpoint_is_reusable`: decide la reutilización comparando contra el `_config.json`, no contra
  la ruta. Un ajuste ausente del nombre del fichero se heredaría en silencio.

### `src/eval_expclr.py`

Camino de evaluación propio de E3, separado de `downstream.py` porque este tiene cuatro defectos
documentados en su docstring. Piezas:

- `extract_embeddings`: una sola pasada en `eval()` bajo `no_grad`.
- `fit_probe`: `StandardScaler` más `Ridge`, con la penalización elegida por validación cruzada
  interna **agrupada por sujeto**. Al ser forma cerrada no depende de learning rate ni de épocas.
- `fit_knn_probe`: la sonda de 1 vecino. El paper lee la **diferencia** entre la lineal y la KNN
  como diagnóstico de si la representación forma clústeres o solo es linealmente explotable (p. 6).
- `aggregate_to_sessions`: mediana por `(subject, age, block)`. La unidad es la **sesión**, que es
  donde el target está definido.
- `session_metrics` y `subject_metrics`: las primeras son la cifra principal; las segundas
  reproducen `master_table.py` para poder comparar con E0-E2, aunque promedien visitas.
- `bootstrap_ci` y `paired_bootstrap_difference`: remuestrean **sujetos**, no sesiones, porque las
  sesiones de un mismo niño no son independientes.

### `run_e3_loso.py`

Ejecuta LOSO evaluando todos los métodos **por el mismo camino**: mismos folds, mismo probe, misma
agregación, misma métrica. Es lo que hace informativa la comparación.

| Id | Método | Qué descarta |
|---|---|---|
| ExpCLR | encoder preentrenado con la pérdida | — |
| B0 | media del entrenamiento | suelo real |
| B1 | Ridge sobre el descriptor crudo | circularidad |
| B2 | encoder con pesos aleatorios | que baste una proyección no lineal |
| B3 | ExpCLR con el descriptor permutado | que importe la geometría y no la forma de la pérdida |
| B7 | Ridge sobre `[embedding ; descriptor]` | aporte incremental del embedding |

`EmbeddingCache` extrae una vez por encoder: ExpCLR y B7 comparten el suyo dentro del fold, y el de
B2 es idéntico en todos los folds porque nunca se entrena.

### `tune_expclr.py`

Elige `Delta`, `lr` y `sim_max` sobre un **80/20 por sujeto**, siguiendo el procedimiento del paper
(p. 6) de ajustar en un split reservado y comparar en el conjunto completo. Dos etapas para que
quepan 19 configuraciones en lugar de 96. Los sujetos de validación se **excluyen** del LOSO
posterior, de modo que la cifra final no está contaminada por la selección.

---

## 6. Desviaciones declaradas respecto al paper

### Inevitables por el dominio

| Aspecto | Paper | Aquí |
|---|---|---|
| Tarea | clasificación, 4 a 6 clases, accuracy | regresión de edad, MAE |
| Evaluación | split fijo por muestra | LOSO, agregación a sesión, bootstrap de sujetos |
| Ajuste de hiperparámetros | 80/20 por muestra | 80/20 por sujeto (por muestra sería fuga) |
| Descriptor | crudo | z-score, forzoso con desviaciones de 4.1e-12 a 28.5 |
| Encoder | TCN y ResNet, 8 bloques | CNN 2D, BiLSTM y atención, heredado |
| Tamaño | 7.352 a 59.922 muestras | 2.609 ventanas, 274 sesiones, 45 sujetos |

El cambio de encoder está autorizado expresamente (*"ExpCLR is not restricted to this
architecture"*, p. 6), y la condición que el paper sí impone, misma arquitectura para todos los
métodos comparados, se cumple.

### Elecciones propias

- **`Delta` ajustado** en lugar de fijado a 1, y **`tau` explorado** en lugar de dejado en 1. Sobre
  `Delta`, ver sección 4: el valor usado cae dentro de la meseta que el paper valida. Sobre `tau`,
  los propios autores declaran que su elección no es óptima: *"other tau < 1.0 could improve the
  performance slightly"* (App. B.4), y el panel derecho de su figura 2 muestra la precisión bajando
  de forma monótona de `tau = 0.5` a `tau = 100`.
- **`dropout = 0`** durante el preentrenamiento. En el paper el dropout aparece solo como
  augmentation del baseline SimCLR, nunca como regularizador de ExpCLR, cuya tesis es que no usa
  transformaciones. Dejarlo activo haría además `E(x_i)` estocástico justo donde la pérdida mide
  distancias.
- **Dimensión 128** en vez de 100. Cae en la meseta plana de la curva de sensibilidad del paper.
- **B0, B3 y B7** son baselines propios que el paper no tiene. B3 en particular es un control de
  falsabilidad genuino.

### Pendiente

Faltan tres baselines de la Tabla 2 que serían informativos. El más valioso es **Expert Feature
Decoding**, que es la Proposición 1 hecha experimento y aísla si la pérdida contrastiva aporta algo
sobre simplemente regresionar el descriptor. El paper lo describe con detalle suficiente para
replicarlo (Sec. 4.3, p. 7): *"the embedding is given by the output of the penultimate layer of a
network that is trained by learning to predict the expert features from the raw time-series. During
training we minimized the MSE-loss and add a projection layer to the architecture used by the other
methods"*. Es decir, la arquitectura común más una capa de proyección extra a las `d` dimensiones
del descriptor, entrenada por MSE, evaluando la penúltima capa. B1 no lo sustituye porque no usa
encoder y por tanto no dice si la señal cruda contiene la información.

Esa misma frase, de paso, confirma que la arquitectura común **no tiene** una cabeza descartable:
la capa de proyección se añade *"to the architecture used by the other methods"* solo para este
baseline.

Faltan también **SimCLR** por el mismo camino de evaluación, un techo supervisado, y ejecutar con
varias semillas: el paper reporta media y error estándar sobre cinco ensayos, y aquí se corre una
sola.

---

## 7. Cómo ejecutarlo

**1. Construir el descriptor** alineado al orden de ventanas:

```bash
conda run -n dasci-cimcyc python ../src/build_expert_features.py \
    --meta_path  data/processed/all_all/processed_metadata.csv \
    --output_dir data/processed/expert_features \
    --descriptor P_diverso
```

**2. Elegir los hiperparámetros** en el split reservado:

```bash
conda run -n dasci-cimcyc python -W ignore -u tune_expclr.py --epochs 30
```

Escribe `save/e3_tuning/best_config.json` con `delta`, `lr`, `sim_max` y los sujetos de validación.

**3. Ejecutar la malla LOSO**, leyendo esa configuración para no transcribirla a mano:

```bash
conda run -n dasci-cimcyc python -W ignore -u run_e3_loso.py \
    --epochs 50 --config save/e3_tuning/best_config.json
```

Coste aproximado: 3 minutos de preentrenamiento por fold.

**Entrenar un solo encoder** (para inspeccionar o depurar):

```bash
conda run -n dasci-cimcyc python -u src/train_expclr.py \
    --data_path       data/processed/all_all \
    --expert_features data/processed/expert_features/expert_features_P_diverso.npy \
    --descriptor P_diverso --zone all --frequency all \
    --delta 1.86 --lr 0.005 --num_epochs 50 --diagnose_every 5 \
    --exclude_subjects B010
```

Durante el entrenamiento conviene mirar **la dimensionalidad efectiva, no solo la pérdida**. Si cae,
el encoder está contrayendo la representación y avisa por consola; si la pérdida se pega al suelo
equidistante, `Delta` no está calibrado al descriptor.

Ablaciones disponibles: `--no_hard_negative_mining` (Eq. 3 en lugar de Eq. 4), `--linear_similarity`
(Eq. 2 en lugar de Eq. 5), `--sim_max batch`, `--loss_on embedding` (encoder dos capas más corto),
`--temperature` (el minado de negativos duros) y `--dropout 0.25` (el valor heredado).

Nota sobre el uso: el paper contempla explícitamente la aplicación que se hace aquí. *"ExpCLR can
also be applied to supervised CL with datasets containing continuous labels, e.g. regression tasks
such as pose estimation"* (p. 9).

## Tests

```bash
conda run -n dasci-cimcyc python -m pytest tests/ -q
```

Los que fijan las propiedades descritas aquí: `test_row_mean_of_normalised_distance_is_always_one`,
`test_equidistant_geometry_sits_at_n_over_n_minus_one_not_at_one`,
`test_loss_is_invariant_to_rescaling_the_embedding`, `test_zero_loss_is_unreachable_at_delta_one` y
`test_checkpoint_with_a_different_loss_target_is_not_reused`.
