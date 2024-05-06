import blendertoolbox as bt
import bpy
import os
cwd = os.getcwd()

def setMat_vertex_color(mesh):
	mat = bpy.data.materials.new('MeshMaterial')
	mesh.data.materials.append(mat)
	mesh.active_material = mat
	mat.use_nodes = True
	tree = mat.node_tree


	vcolor = tree.nodes.new('ShaderNodeVertexColor')
	tree.links.new(vcolor.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])
	# set principled BSDF
	tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = 0.3
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
        RGBA = (160.0/255, 160.0/255, 160.0/255, 1)
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