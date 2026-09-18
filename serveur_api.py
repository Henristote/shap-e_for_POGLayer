import torch
import trimesh
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

# Définition du format de la requête attendue depuis Unity
class RequeteGeneration(BaseModel):
    prompt: str

# Initialisation de l'API
app = FastAPI()

# Chargement global des modèles (exécuté une seule fois au démarrage du serveur)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

@app.post("/generer")
async def generer_objet(requete: RequeteGeneration):
    prompt = requete.prompt
    batch_size = 1
    guidance_scale = 15.0

    # Lancement de la génération sur la carte graphique
    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=False,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    # Sauvegarde de l'objet au format GLB
    nom_fichier = requete.prompt.replace(" ", "_")
    t = decode_latent_mesh(xm, latents[0]).tri_mesh()
    with open(f'{nom_fichier}.ply', 'wb') as f:
        t.write_ply(f)
    mesh = trimesh.load(f'{nom_fichier}.ply')
    rotation_matrix = trimesh.transformations.rotation_matrix(
        np.radians(-90), [1, 0, 0]
    )
    mesh.apply_transform(rotation_matrix)
    mesh.export(f'{nom_fichier}.glb')
    
    # Retourne le fichier 3D directement à l'application appelante
    return FileResponse(f'{nom_fichier}.glb', media_type='application/octet-stream', filename=f'{nom_fichier}.glb')