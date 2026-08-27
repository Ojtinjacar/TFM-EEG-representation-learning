# Resultados consolidados

Base única para los notebooks. Se genera desde la raíz del repositorio con:

```
python src/consolidate_results.py --source save
```

No se edita a mano: cuando cierre cada corrida se vuelve a ejecutar y se sobrescribe.

## Qué hay

| Fichero | Contenido |
|---|---|
| `results_folds.csv` | una fila por fold |
| `results_subjects.csv` | una fila por sujeto y fold, con `y_true` e `y_pred` |
| `by_method/<variante>/<zona>.csv` | lo mismo que la tabla de folds, partido por método |
| `MANIFEST.json` | procedencia, md5 de cada fichero de origen, commit y recuentos |

## El aviso que hay que leer antes de calcular nada

**No uses la columna `r2` de `results_folds.csv`.**

Esa columna es lo que escribió el pipeline, y **el pipeline calcula la métrica dentro de
cada sujeto y luego promedia**. Con un solo registro por sujeto la etiqueta es constante,
el R2 intra-sujeto degenera y salen valores absurdos. El mejor modelo del trabajo,
`SimCLR-xsubj-wasser` en occipital, da **+0,048** promediando el R2 por fold y **+0,817**
agrupando los 45 sujetos. Son las mismas predicciones.

La cifra que reporta la memoria es el **R2 agrupado sobre los sujetos**, que se calcula
desde `results_subjects.csv`:

```python
import pandas as pd
s = pd.read_csv("results/results_subjects.csv")

def r2_agrupado(d):
    y, p = d.y_true.values, d.y_pred.values
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

(s[(s.target == "age") & (s.eval_mode == "linear_probe")]
   .groupby(["variant", "zone"]).apply(r2_agrupado))
```

El RMSE del pipeline sí es utilizable: es el que reproduce las cifras publicadas del paper.

## Columnas

`family` el método base; `variant` la etiqueta completa; `strategy` el criterio de vecino
(`nbr`, `xsubj`, `diffage`, `lag0`, y `augment` para el control); `metric` la distancia
(`cosine`, `wasserstein`, `riemann`); `descriptor` el descriptor experto de ExpCLR; `zone`,
`frequency`, `target`, `eval_mode`, `fold`; `campaign` la corrida de la que salió y
`source_file` el fichero exacto.

**No hay columna con la máquina, y es deliberado.** El trabajo se reparte entre varias VMs
por método, pero eso es un detalle operativo: la misma campaña ejecutada en una máquina o en
cuatro tiene que dar exactamente la misma base. Tampoco hay `canonical`: con un directorio
por corrida ya no existen procedencias duplicadas que desempatar.

## Qué entra y qué no

**Entra** solo lo que tiene sus **10 folds**. Una corrida a medias no es una versión más
débil de una terminada: agrupada da una cifra sobre una fracción de la cohorte que se lee
como si fuera la real. El consolidador avisa por pantalla de lo que descarta y lo registra
en `incomplete_runs_dropped` del manifiesto.

**No entra** nada que estuviera en ejecución en el momento de consolidar.

## Estado

Esta base sale de la campaña reejecutada desde cero, no de la anterior. Hoy contiene las dos
corridas que han cerrado:

| Corrida | Métodos | Regiones | Folds |
|---|---|---|---|
| `expclr` | `ExpCLR`, `ExpCLR-full`, `ExpCLR-aper` | las cinco (`P_aper` solo cabeza completa) | 110 |
| `baselines_paper` | `PCA`, `AE`, `MAE`, `TripletLoss`, `supervised` | cabeza completa | 50 |

Quedan por incorporar las trece variantes de SimCLR en las cinco regiones, el VAE e
InterFusion. Se añaden reejecutando el consolidador cuando sus corridas cierren; la
disposición no cambia, solo aparecen más carpetas dentro de `save/`.
