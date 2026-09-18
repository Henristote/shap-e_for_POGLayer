# Shap-E API Server for 3D Generation

This project offers a lightweight, high-performance API built with FastAPI to leverage OpenAI's Shap-E model. It enables the dynamic generation of 3D models from text descriptions and their export in the standard OBJ format. The architecture is designed to offload computationally intensive tasks—which require an NVIDIA graphics card—away from a thin client, such as a standalone virtual reality headset running a Unity application.

## Setting up the work environment

To avoid conflicts with your operating system's global libraries and to comply with Linux security standards (specifically the `externally-managed-environment` restriction), it is essential to isolate the installation.

### On Ubuntu / Debian Linux:

If the virtual environment management tool is not yet installed on your system, install it first using the `apt` package manager:

```
sudo apt update
sudo apt install python3-venv python3-pip

```

Next, navigate to the root of this directory in your terminal and create a fresh virtual environment named `venv`:

```
python3 -m venv venv

```

Then, simply activate this workspace to isolate your installations within it. If you are working in a Linux or macOS environment, use the command:

```
source venv/bin/activate

```

On a Windows system, the syntax is:

```
venv\Scripts\activate

```

The appearance of the environment name in parentheses at the beginning of your command line confirms that the operation was successful.

## Installing Dependencies

Now that your environment is active, you can proceed to download the necessary packages.

The top priority is installing the PyTorch library with support for NVIDIA hardware acceleration (via CUDA 12.1). This component ensures optimal generation times on your dedicated graphics card:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

```

Next, install the Shap-E project's internal requirements by performing a local installation from the current directory. This action scans the folder's configuration files and fetches the associated dependencies:

```
pip install -e .

```

> **Note:** If the Python interpreter reports a missing text-parsing module upon the first launch (`ModuleNotFoundError: No module named 'yaml'`), you can resolve this by simply running `pip install pyyaml`.

Complete the setup by adding FastAPI, which will structure our server, and Uvicorn, which will serve as the web execution engine:

```
pip install fastapi uvicorn

```

## Starting the local server

The main script (`serveur_api.py`) is optimized to load the various OpenAI diffusion models into video memory (VRAM) just once during initialization. This allows the server to remain on standby and respond instantly to requests from Unity without needing to reload hundreds of millions of parameters each time.

To start the server, run the following command:

```
python -m uvicorn serveur_api:app --host 0.0.0.0 --port 5000

```

Your console will display a few confirmation lines indicating that the API is online and listening for instructions on port 5000 of your local machine.

## Exposing to the external network via Cloudflare

If you want this Python server to communicate with an application running on a standalone headset, you may encounter issues with your router's security settings. To bypass this problem without altering your home or office network configuration, setting up a Cloudflare tunnel is the ideal solution.

### Installing Cloudflared on Ubuntu:

Add the official Cloudflare repository to your system, then install it using `apt`:

```
sudo mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main" | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt-get update && sudo apt-get install cloudflared

```

Next, open a second terminal—leaving your Uvicorn server running in the background in the first—and run the exposure command:

```
cloudflared tunnel --url http://localhost:5000

```

The tool will analyze your connection and generate a secure, temporary public address (typically in the format `https://your-tunnel.trycloudflare.com`). You will need to copy this specific URL and integrate it into the C# script in your Unity project. From that point on, voice requests from your game will travel across the Internet to reach this tunnel, triggering the generation of the 3D mesh directly on your host computer.

#

#

#

#

# Shap-E (original README)

This is the official code and model release for [Shap-E: Generating Conditional 3D Implicit Functions](https://arxiv.org/abs/2305.02463).

- See [Usage](#usage) for guidance on how to use this repository.
- See [Samples](#samples) for examples of what our text-conditional model can generate.

# Samples

Here are some highlighted samples from our text-conditional model. For random samples on selected prompts, see [samples.md](samples.md).

<table>
    <tbody>
        <tr>
            <td align="center">
                <img src="samples/a_chair_that_looks_like_an_avocado/2.gif" alt="A chair that looks like an avocado">
            </td>
            <td align="center">
                <img src="samples/an_airplane_that_looks_like_a_banana/3.gif" alt="An airplane that looks like a banana">
            </td align="center">
            <td align="center">
                <img src="samples/a_spaceship/0.gif" alt="A spaceship">
            </td>
        </tr>
        <tr>
            <td align="center">A chair that looks<br>like an avocado</td>
            <td align="center">An airplane that looks<br>like a banana</td>
            <td align="center">A spaceship</td>
        </tr>
        <tr>
            <td align="center">
                <img src="samples/a_birthday_cupcake/3.gif" alt="A birthday cupcake">
            </td>
            <td align="center">
                <img src="samples/a_chair_that_looks_like_a_tree/2.gif" alt="A chair that looks like a tree">
            </td>
            <td align="center">
                <img src="samples/a_green_boot/3.gif" alt="A green boot">
            </td>
        </tr>
        <tr>
            <td align="center">A birthday cupcake</td>
            <td align="center">A chair that looks<br>like a tree</td>
            <td align="center">A green boot</td>
        </tr>
        <tr>
            <td align="center">
                <img src="samples/a_penguin/1.gif" alt="A penguin">
            </td>
            <td align="center">
                <img src="samples/ube_ice_cream_cone/3.gif" alt="Ube ice cream cone">
            </td>
            <td align="center">
                <img src="samples/a_bowl_of_vegetables/2.gif" alt="A bowl of vegetables">
            </td>
        </tr>
        <tr>
            <td align="center">A penguin</td>
            <td align="center">Ube ice cream cone</td>
            <td align="center">A bowl of vegetables</td>
        </tr>
    </tbody>
<table>

# Usage

Install with `pip install -e .`.

To get started with examples, see the following notebooks:

- [sample_text_to_3d.ipynb](shap_e/examples/sample_text_to_3d.ipynb) - sample a 3D model, conditioned on a text prompt.
- [sample_image_to_3d.ipynb](shap_e/examples/sample_image_to_3d.ipynb) - sample a 3D model, conditioned on a synthetic view image. To get the best result, you should remove background from the input image.
- [encode_model.ipynb](shap_e/examples/encode_model.ipynb) - loads a 3D model or a trimesh, creates a batch of multiview renders and a point cloud, encodes them into a latent, and renders it back. For this to work, install Blender version 3.3.1 or higher, and set the environment variable `BLENDER_PATH` to the path of the Blender executable.
