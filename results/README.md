# Resultados consolidados

Base única para los notebooks. Se genera con:

```
python src/consolidate_results.py --source <dir con <maquina>/<ruta save>/*.csv>
```

No se edita a mano: cuando cierren las corridas en vuelo se vuelve a ejecutar.

## Qué hay

| Fichero | Contenido |
|---|---|
| `results_folds.csv` | una fila por fold |
| `results_subjects.csv` | una fila por sujeto y fold, con `y_true` e `y_pred` |
| `by_method/<variante>/<zona>.csv` | lo mismo que la tabla de folds, partido por método |
| `MANIFEST.json` | procedencia, md5 del fichero de origen, commit y recuento |

## Dos avisos que hay que leer antes de calcular nada

### 1. No uses la columna `r2` de `results_folds.csv`

Esa columna es lo que escribió el pipeline, y **el pipeline calcula la métrica dentro de
cada sujeto y luego promedia**. Con un solo registro por sujeto la etiqueta es constante,
el R2 intra-sujeto degenera y salen valores absurdos: hay folds con R2 de -14.

La cifra que reporta la memoria es el **R2 agrupado sobre los sujetos**, que se calcula
desde `results_subjects.csv`:

```python
import pandas as pd
s = pd.read_csv("results/results_subjects.csv")
s = s[s.canonical]

def r2_agrupado(d):
    y, p = d.y_true.values, d.y_pred.values
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

(s[(s.target == "age") & (s.eval_mode == "linear_probe")]
   .groupby(["variant", "zone"]).apply(r2_agrupado))
```

El RMSE del pipeline sí es utilizable: es el que reproduce las cifras publicadas del paper.

### 2. Filtra por `canonical`

La cabeza completa se corrió en varias campañas, así que 24 combinaciones existen entre dos
y cuatro veces. Todas son reales, y una de esas parejas es justo lo que mide la dispersión
de una reejecución. Pero agrupar por método y zona sin filtrar **cuenta el mismo sujeto
varias veces** y lo lee como una cohorte mayor.

`canonical=True` deja una procedencia por corrida, eligiendo la campaña que cubre más zonas,
que es la que sirve para comparar entre regiones.

## Columnas

`family` el método base; `variant` la etiqueta completa **sin la zona**; `strategy` el
criterio de vecino (`nbr`, `xsubj`, `diffage`, `lag0`, `augment` para el control);
`metric` la distancia (`cosine`, `wasserstein`, `riemann`); `descriptor` el descriptor
experto de ExpCLR; `zone`, `frequency`, `target`, `eval_mode`, `fold`; `campaign`, `machine`
y `source_dir` para la trazabilidad.

## Qué entra y qué no

**Entra** solo lo que tiene sus **10 folds**. Una corrida a medias no es una versión más
débil de una terminada: agrupada da una cifra sobre una fracción de la cohorte que se lee
como si fuera la real.

**No entra**:

- **ExpCLR por zona de la campaña antigua**, que comparaba contra el descriptor de las
  cuatro regiones mientras el encoder veía los canales de una. Está sustituida por la
  corrida con descriptor por zona.
- **`lag0` en occipital**: solo existe el fold 0, porque esa variante se retiró después de
  arrancar la zona. Aparece en el aviso del consolidador y se descarta.
- Lo que estaba corriendo en el momento de la extracción.

## Convenciones del protocolo

Validación cruzada de 10 folds **por sujeto**, semilla 1234 y semilla de fold igual a
`1234 + indice`. 45 sujetos con edad; 39 con `cit_36mo`.

**Diferencias de R2 por debajo de tres centésimas no son interpretables.** Está medido: la
cabeza completa se corrió dos veces con configuración idéntica y dio 0,794 y 0,775, y las
tres métricas de distancia se movieron entre 0,016 y 0,026, incluida `riemann`, que no
depende del montaje. Es el ruido de reejecución en GPU, donde cuDNN no es determinista.
