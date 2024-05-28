import blendertoolbox as bt
import bpy
import os
import torch
from shap_e.models.download import load_model
from shap_e.util.notebooks import decode_latent_mesh

cwd = os.getcwd()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setMat_vertex_color(mesh, roughness: float=0.3):
	mat = bpy.data.materials.new('MeshMaterial')
	mesh.data.materials.append(mat)
	mesh.active_material = mat
	mat.use_nodes = True
	tree = mat.node_tree


	vcolor = tree.nodes.new('ShaderNodeVertexColor')
	tree.links.new(vcolor.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])
	# set principled BSDF
	tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = roughness
	tree.nodes["Principled BSDF"].inputs['Sheen Tint'].default_value = [0, 0, 0, 1]
	tree.nodes["Principled BSDF"].inputs['Specular IOR Level'].default_value = 0.5
	tree.nodes["Principled BSDF"].inputs['IOR'].default_value = 1.45
	tree.nodes["Principled BSDF"].inputs['Transmission Weight'].default_value = 0
	tree.nodes["Principled BSDF"].inputs['Coat Roughness'].default_value = 0

def good_looking_render(meshPath, 
                        outputPath, 
                        render_guidance = False, 
                        shade_smooth = True, 
                        subdivide = False, 
                        plastic = True, 
                        save_file = False,
                        tint = False,
                        resolution_x = 720,
                        resolution_y = 720,
                        num_samples = 100,
                        exposure = 1.5,
                        z_rotation = -140):
    ## initialize blender
    imgRes_x = resolution_x # recommend > 1080 
    imgRes_y = resolution_y # recommend > 1080 
    numSamples = num_samples # recommend > 200
    #imgRes_x = 500 # recommend > 1080 
    #imgRes_y = 500 # recommend > 1080 
    #numSamples = 70 # recommend > 200
    use_GPU = True
    bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure, use_GPU)

    ## read mesh
    location = (-0.074, -0.045, 0.7) # (GUI: click mesh > Transform > Location)
    rotation = (0, 0, z_rotation) # (GUI: click mesh > Transform > Rotation)
    scale = (0.6,0.6,0.6) # (GUI: click mesh > Transform > Scale)
    mesh = bt.readMesh(meshPath, location, rotation, scale)

    ## set shading (uncomment one of them)
    if shade_smooth:
        bpy.ops.object.shade_smooth()
    else:
        bpy.ops.object.shade_flat()
    
    ## subdivision
    if subdivide:
        bt.subdivision(mesh, level = 1)
    
    ###########################################
    ## Set your material here (see other demo scripts)
    print("NOT SETTING PLASTIC")
    if plastic:
        RGBA = (130.0/255, 130.0/255, 130.0/255, 1)
        #RGBA = (130.0/255, 100.0/255, 100.0/255, 1)
        if tint:
            RGBA = (100.0/255, 130.0/255, 130.0/255, 1)
        meshColor = bt.colorObj(RGBA, 0.5, 1.0, 1.0, 0.0, 2.0)
        bt.setMat_plastic(mesh, meshColor)

    if not render_guidance:
        setMat_vertex_color(mesh)

    ## End material
    ###########################################

    ## set invisible plane (shadow catcher)
    bt.invisibleGround(shadowBrightness=0.9)

    ## set camera 
    ## Option 1: don't change camera setting, change the mesh location above instead
    camLocation = (3, 0, 2)
    lookAtLocation = (0,0,0.5)
    focalLength = 45 # (UI: click camera > Object Data > Focal Length)
    cam = bt.setCamera(camLocation, lookAtLocation, focalLength)

    ## set light
    lightAngle = (6, -30, -155) 
    strength = 7.0
    shadowSoftness = 0.3
    sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)

    ## set ambient light
    bt.setLight_ambient(color=(0.25,0.25,0.25,1)) 

    ## set gray shadow to completely white with a threshold (optional but recommended)
    bt.shadowThreshold(alphaThreshold = 0.05, interpolationMode = 'CARDINAL')

    ## save blender file so that you can adjust parameters in the UI
    if save_file:
        bpy.ops.wm.save_mainfile(filepath=os.getcwd() + '/test.blend')

    ## Make High Contrast
    bpy.data.scenes['Scene'].render.image_settings.color_management = 'OVERRIDE'
    bpy.context.scene.view_settings.look = 'AgX - Medium High Contrast'

    ## save rendering
    bt.renderImage(outputPath, cam)

# make this script executable with argparse
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Render a mesh using blender')
    parser.add_argument('mesh_path', type=str, help='Path to the mesh file')
    parser.add_argument('output_path', type=str, help='Path to the output image')
    parser.add_argument('--from_latent', action='store_true', help='Render mesh from latent')
    parser.add_argument('--render_guidance', action='store_true', help='Render guidance')
    parser.add_argument('--tint', action='store_true', help='Render guidance with tint')
    parser.add_argument('--shade_smooth', action='store_true', help='Shade smooth')
    parser.add_argument('--subdivide', action='store_true', help='Subdivide')
    parser.add_argument('--plastic', action='store_true', help='Plastic')
    parser.add_argument('--save_file', action='store_true', help='Save file')
    parser.add_argument('--resolution_x', type=int, default=720, help='Resolution x')
    parser.add_argument('--resolution_y', type=int, default=720, help='Resolution y')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples')
    parser.add_argument('--exposure', type=float, default=1.5, help='Exposure')
    parser.add_argument('--z_rotation', type=float, default=-140, help='Z rotation')
    args = parser.parse_args()
    # make output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    mesh_path = args.mesh_path
    if args.from_latent:
        xm = load_model('transmitter', device=device)
        latent = torch.load(args.mesh_path)
        t = decode_latent_mesh(xm, latent).tri_mesh()
        mesh_path = os.path.join(args.output_path, "mesh.ply")
        with open(mesh_path, 'wb') as f:
            t.write_ply(f)
    good_looking_render(mesh_path, 
                        args.output_path, 
                        args.render_guidance, 
                        args.shade_smooth, 
                        args.subdivide, 
                        args.plastic, 
                        args.save_file, 
                        args.tint, 
                        args.resolution_x, 
                        args.resolution_y, 
                        args.num_samples, 
                        args.exposure, 
                        args.z_rotation)