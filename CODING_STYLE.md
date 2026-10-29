# Estilo de programación de Marco

Extraído de `rsm_design` (2023–2024), un proyecto de análisis científico escrito a mano,
antes de programar con asistentes de IA. Sirve como referencia de estilo para nuevos
proyectos en Python y como instrucciones para asistentes de IA, para que el código que
generen se lea como el mío.

## 1. Estructura del proyecto
- **Librería + scripts delgados.** La lógica reutilizable vive en una carpeta de módulos
  (`utils/rsm.py`, `utils/plots.py`, `utils/univariate.py`); cada análisis es un script
  corto en la raíz que define parámetros y llama a esas funciones.
- **Un script = un análisis = una carpeta de resultados** (`rsm_full.py` → `rsm_results_full/`).
- **Los resultados son archivos**: CSV para números, TXT para resúmenes estadísticos,
  PNG/JPEG para figuras, todo en la carpeta de resultados.
- El código reemplazado se mueve a `old_scripts/` antes de borrarlo.

## 2. Scripts
- Docstring de módulo al inicio que dice qué hace el script.
- Parámetros como variables simples justo después de los imports, con raw strings para rutas:
  ```python
  experiments_file = r"experimental_results.xlsx"
  results_folder = r"rsm_results_full"
  sheet_name = "RSM_3"
  factors = ["Extractant-Dispersant Ratio", "Sample volume", "Extraction mixture volume"]
  ```
- Las funciones propias del script van arriba de una línea separadora `####...####`;
  el código "principal" va debajo (sin `if __name__ == "__main__":`).
- Las llamadas usan **argumentos con nombre**, uno por línea.
- Las variantes de un modelo se prueban **comentando y descomentando elementos de una lista**,
  con un `#` corto como marca en los que se están evaluando:
  ```python
  del_coeff=[
      # "b0",
      # "b12",  #
      # "b13",  #
  ],
  ```

## 3. Funciones
- Nombres en `snake_case` que dicen la acción: `extract_data`, `model_fitting`,
  `import_coeff`, `rsm_optimization`, `residual_plot`, `coeff_plot`.
- Docstring en cada función de librería, con este formato:
  ```python
  """Descripción corta de qué hace la función y por qué.

  params:
      name (type): significado
  return:
      name (type): significado
  """
  ```
- El razonamiento del dominio va en el docstring (por ejemplo, por qué el óptimo de un
  CCD debe restringirse al elipsoide de los puntos experimentales).
- Comportamiento opcional mediante flags con valores por defecto
  (`normalize=None`, `whiskers="ci"`, `folder=None`).
- Patrón para figuras: **guardar si hay carpeta, si no mostrar**:
  ```python
  if folder != None:
      plt.savefig(os.path.join(folder, file_name), dpi=DPI)
  else:
      plt.show()
  ```
- Nombres de archivo derivados de la respuesta: `f'{response.replace(" ", "")}_coeff_plot.png'`.

## 4. Cuerpo del código
- El código se divide en **secciones comentadas en mayúsculas**:
  `# PROCESSING DATA`, `# MAKING PLOT`, `# SAVE OR SHOW`, `## PLOT REGRESSION ANALYSIS`
  (`#` para sección, `##` para subsección).
- Comentarios en línea cortos que explican el *significado*, no la sintaxis:
  `resd = y_exp - y_pred  # Residual`, `data_df = data_df.drop([0, 1])  # drop units rows`.
- Las transformaciones paso a paso reasignan la misma variable (`data_df = ...` varias
  veces) en vez de encadenar métodos largos.
- Nombres de variables que mezclan notación del dominio y nombres descriptivos:
  `X`, `y`, `y_exp`, `y_pred`, `resd`, `coeff`, `data_df`, `centre_point`,
  `factor_bounds`. Los DataFrames terminan en `_df`.
- Constantes de módulo en MAYÚSCULAS al inicio (`DPI = 600`); "constantes" locales
  marcadas con comentario (`grid_space = 100  # CONSTANT`).
- Imports: primero la librería estándar, luego `import x as y` para el stack científico
  (`numpy as np`, `pandas as pd`, `statsmodels.api as sm`, `matplotlib.pyplot as plt`),
  y al final las líneas `from x import y`.
- Formato compatible con **Black** (comillas dobles, comas finales, 88 columnas).
- Código e identificadores en inglés.

## 5. Salida por consola
- Cada función se anuncia y reporta sus pasos con líneas indentadas con tab,
  de modo que una ejecución se lee como una bitácora de laboratorio:
  ```text
  Fitting model to Recovery response.
      Saving model coefficient.
  residual_plot()
      Mean Squared Error (MSE): ...
  ```

## 6. Stack científico y decisiones
- pandas para datos tabulares leídos desde Excel (`read_excel`, una hoja por experimento).
- statsmodels `OLS` para la regresión porque entrega la tabla resumen/ANOVA completa;
  scikit-learn solo para utilidades (`PolynomialFeatures`, `mean_squared_error`).
- scipy para optimización (`minimize`, L-BFGS-B con bounds) y estadística.
- matplotlib con objetos `fig, ax` explícitos, `gridspec` para figuras compuestas,
  `ax.grid(True)` y exportación en alta resolución (600–800 DPI) lista para publicar.
- Los diagnósticos estadísticos siempre se grafican: coeficientes con IC, residuos con
  histograma + curva normal, predicho vs. experimental.

## 7. Git
- Commits pequeños y frecuentes, con mensajes cortos en inglés simple
  (`now rsm is in utils`, `Sigma profile plots`, `pycache ignored`).
- `.gitignore` para entornos y cachés (`venv/`, `__pycache__/`, `.vscode/`).

## 8. Mantener vs. evolucionar (para proyectos nuevos)
**Mantener:** librería + scripts delgados, docstrings con razonamiento del dominio,
código por secciones con comentarios con significado, llamadas con argumentos con nombre,
figuras "guardar o mostrar", salida tipo bitácora, statsmodels para inferencia,
gráficos con calidad de publicación.

**Evolucionar:**
- Usar `is None` / `is True` en vez de `== None` / `== True`.
- Reemplazar `from module import *` por imports explícitos.
- Pasar parámetros por funciones o CLI en vez de fijar en el código rutas, nombres de
  hojas, columnas o número de factores.
- Usar `logging` (o un flag `verbose`) en vez de `print` dentro de la librería;
  devolver resultados además de escribir archivos.
- Cerrar las figuras (`plt.close(fig)`) después de guardarlas.
- Agregar type hints y una carpeta `tests/`; declarar dependencias en `pyproject.toml`.
- Usar `pathlib.Path` en vez de rutas con `\` de Windows.
- Quitar prints de depuración y líneas "DELETE IN FUTURE" antes de hacer commit.
