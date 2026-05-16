from pathlib import Path
import argparse
import json
import re

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd


RUTA_DATOS = "dataset/OpenNeuro/images"
RUTA_SALIDA = "dataset/OpenNeuro/metadata"


def obtener_sujeto(ruta: Path):
    resultado = re.search(r"sub-(RC\d+)", str(ruta))
    if resultado:
        return f"sub-{resultado.group(1)}"
    return None


def obtener_sesion(ruta: Path):
    resultado = re.search(r"ses-\d+", str(ruta))
    if resultado:
        return resultado.group(0)
    return "unknown"


def obtener_grupo(sujeto: str):
    if sujeto is None:
        return None, "unknown"

    if sujeto.startswith("sub-RC41"):
        return 0, "control"

    if sujeto.startswith("sub-RC42"):
        return 1, "parkinson"

    return None, "unknown"


def buscar_imagenes_t1(ruta_base: Path, usar_derivados: bool = True):
    patrones = [
        "sub-*/ses-*/anat/*T1wbrain.nii.gz",
        "sub-*/ses-*/anat/*T1w*.nii.gz",

        "derivatives/sub-*/ses-*/anat/*T1wbrain.nii.gz",
        "derivatives/sub-*/ses-*/anat/*T1w*.nii.gz",

        "**/*T1wbrain.nii.gz",
        "**/*T1w*.nii.gz",
    ]

    imagenes = []
    for patron in patrones:
        imagenes.extend(ruta_base.glob(patron))

    return sorted(set(imagenes))


def calcular_datos_imagen(ruta_imagen: Path):
    try:
        imagen = nib.load(str(ruta_imagen))
        volumen = np.asanyarray(imagen.dataobj)

        total_voxeles = int(volumen.size)
        voxeles_utiles = int(np.count_nonzero(volumen))

        if np.issubdtype(volumen.dtype, np.floating):
            tiene_nan = bool(np.isnan(volumen).any())
        else:
            tiene_nan = False

        return {
            "estado": "OK",
            "dimensiones": str(volumen.shape),
            "num_dimensiones": int(volumen.ndim),
            "tipo_dato": str(volumen.dtype),
            "minimo": float(np.nanmin(volumen)),
            "maximo": float(np.nanmax(volumen)),
            "media": float(np.nanmean(volumen)),
            "desviacion": float(np.nanstd(volumen)),
            "tiene_nan": tiene_nan,
            "voxeles_no_cero": voxeles_utiles,
            "voxeles_totales": total_voxeles,
            "porcentaje_no_cero": float(voxeles_utiles / total_voxeles) if total_voxeles > 0 else 0.0,
            "error": "",
        }

    except Exception as error:
        return {
            "estado": "ERROR",
            "dimensiones": "",
            "num_dimensiones": "",
            "tipo_dato": "",
            "minimo": "",
            "maximo": "",
            "media": "",
            "desviacion": "",
            "tiene_nan": "",
            "voxeles_no_cero": "",
            "voxeles_totales": "",
            "porcentaje_no_cero": "",
            "error": str(error),
        }


def guardar_vista_previa(ruta_imagen: Path, ruta_salida: Path):
    try:
        imagen = nib.load(str(ruta_imagen))
        volumen = np.asanyarray(imagen.dataobj).astype(np.float32)

        if volumen.ndim != 3:
            return False

        mascara = volumen > 0
        if np.any(mascara):
            p1, p99 = np.percentile(volumen[mascara], [1, 99])
        else:
            p1, p99 = volumen.min(), volumen.max()

        volumen = np.clip(volumen, p1, p99)

        if volumen.max() > volumen.min():
            volumen = (volumen - volumen.min()) / (volumen.max() - volumen.min())

        corte_x = volumen.shape[0] // 2
        corte_y = volumen.shape[1] // 2
        corte_z = volumen.shape[2] // 2

        plano_sagital = np.rot90(volumen[corte_x, :, :])
        plano_coronal = np.rot90(volumen[:, corte_y, :])
        plano_axial = np.rot90(volumen[:, :, corte_z])

        fig, ejes = plt.subplots(1, 3, figsize=(12, 4))

        ejes[0].imshow(plano_sagital, cmap="gray")
        ejes[0].set_title("Sagital")
        ejes[0].axis("off")

        ejes[1].imshow(plano_coronal, cmap="gray")
        ejes[1].set_title("Coronal")
        ejes[1].axis("off")

        ejes[2].imshow(plano_axial, cmap="gray")
        ejes[2].set_title("Axial")
        ejes[2].axis("off")

        plt.tight_layout()
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(ruta_salida, dpi=120)
        plt.close()

        return True

    except Exception as error:
        print(f"No se pudo crear la vista previa de {ruta_imagen}: {error}")
        return False


def guardar_resumenes(tabla: pd.DataFrame, carpeta_salida: Path):
    imagenes_validas = tabla[tabla["estado"] == "OK"].copy()

    resumen_sujetos = (
        imagenes_validas[["sujeto", "grupo", "etiqueta"]]
        .drop_duplicates()
        .groupby("grupo")
        .agg(num_sujetos=("sujeto", "count"))
        .reset_index()
    )

    resumen_imagenes = (
        imagenes_validas
        .groupby("grupo")
        .agg(
            num_imagenes=("ruta_imagen", "count"),
            num_sujetos=("sujeto", "nunique"),
        )
        .reset_index()
    )

    resumen_sesiones = (
        imagenes_validas
        .groupby(["grupo", "sesion"])
        .agg(
            num_imagenes=("ruta_imagen", "count"),
            num_sujetos=("sujeto", "nunique"),
        )
        .reset_index()
    )

    resumen_sujetos.to_csv(carpeta_salida / "subject_summary.csv", index=False)
    resumen_imagenes.to_csv(carpeta_salida / "image_summary.csv", index=False)
    resumen_sesiones.to_csv(carpeta_salida / "session_summary.csv", index=False)

    resumen_general = {
        "ruta_datos": RUTA_DATOS,
        "total_imagenes_encontradas": int(len(tabla)),
        "imagenes_validas": int((tabla["estado"] == "OK").sum()),
        "imagenes_con_error": int((tabla["estado"] == "ERROR").sum()),
        "total_sujetos": int(imagenes_validas["sujeto"].nunique()) if not imagenes_validas.empty else 0,
        "sujetos_por_grupo": resumen_sujetos.to_dict(orient="records"),
        "imagenes_por_grupo": resumen_imagenes.to_dict(orient="records"),
        "sesiones": sorted(imagenes_validas["sesion"].dropna().unique().tolist()) if not imagenes_validas.empty else [],
    }

    with open(carpeta_salida / "dataset_summary.json", "w", encoding="utf-8") as archivo:
        json.dump(resumen_general, archivo, indent=4, ensure_ascii=False)

    return resumen_general, resumen_sujetos, resumen_imagenes


def generar_previews(tabla: pd.DataFrame, carpeta_salida: Path, max_previews: int):
    carpeta_previews = carpeta_salida / "previews"
    imagenes_validas = tabla[tabla["estado"] == "OK"].copy()

    total = 0

    for _, fila in imagenes_validas.iterrows():
        if total >= max_previews:
            break

        nombre_archivo = f"{fila['sujeto']}_{fila['sesion']}_{fila['grupo']}.png"
        ruta_preview = carpeta_previews / nombre_archivo

        creada = guardar_vista_previa(Path(fila["ruta_imagen"]), ruta_preview)

        if creada:
            total += 1

    return total


def explorar_dataset(ruta_datos: Path, carpeta_salida: Path, usar_derivados: bool, max_previews: int):
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    if not ruta_datos.exists():
        raise FileNotFoundError(f"No existe la carpeta indicada: {ruta_datos}")

    print(f"Buscando imágenes anatómicas en: {ruta_datos}")

    imagenes = buscar_imagenes_t1(ruta_datos, usar_derivados=usar_derivados)
    print(f"Imágenes encontradas: {len(imagenes)}")

    registros = []

    for ruta_imagen in imagenes:
        sujeto = obtener_sujeto(ruta_imagen)
        sesion = obtener_sesion(ruta_imagen)
        etiqueta, grupo = obtener_grupo(sujeto)
        datos_imagen = calcular_datos_imagen(ruta_imagen)

        registros.append({
            "sujeto": sujeto,
            "sesion": sesion,
            "grupo": grupo,
            "etiqueta": etiqueta,
            "ruta_imagen": str(ruta_imagen),
            **datos_imagen,
        })

    tabla = pd.DataFrame(registros)

    ruta_indice = carpeta_salida / "openneuro_derivatives_index.csv"
    tabla.to_csv(ruta_indice, index=False)

    resumen_general, resumen_sujetos, resumen_imagenes = guardar_resumenes(tabla, carpeta_salida)
    previews_generadas = generar_previews(tabla, carpeta_salida, max_previews)

    print(f"Índice guardado en: {ruta_indice}")
    print(f"Vistas previas generadas: {previews_generadas}")

    print("\nResumen del dataset")
    print(f"Imágenes totales: {resumen_general['total_imagenes_encontradas']}")
    print(f"Imágenes válidas: {resumen_general['imagenes_validas']}")
    print(f"Sujetos totales: {resumen_general['total_sujetos']}")

    print("\nSujetos por grupo:")
    print(resumen_sujetos.to_string(index=False))

    print("\nImágenes por grupo:")
    print(resumen_imagenes.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, default=RUTA_DATOS)
    parser.add_argument("--out", type=str, default=RUTA_SALIDA)
    parser.add_argument("--raw-images", action="store_true")
    parser.add_argument("--max-previews", type=int, default=20)

    args = parser.parse_args()

    usar_derivados = not args.raw_images

    explorar_dataset(
        ruta_datos=Path(args.raw),
        carpeta_salida=Path(args.out),
        usar_derivados=usar_derivados,
        max_previews=args.max_previews,
    )


if __name__ == "__main__":
    main()