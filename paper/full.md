
# RTR-GS: 3D Gaussian Splatting for Inverse Rendering with Radiance Transfer and Reflection

Yongyang Zhou

Beijing Institute of Technology

Beijing, China

yongyangzhou@bit.edu.cn

Zichen Wang

Beijing Institute of Technology

Beijing, China

zichenwang@bit.edu.cn

Fanglue Zhang

Victoria University of Wellington

Wellington, New Zealand

fanglue.zhang@vuw.ac.nz

Lei Zhang

Beijing Institute of Technology

Beijing, China

State Key Laboratory of Environment Characteristics and

Effects for Near-space

Beijing, China

leizhang@bit.edu.cn

![](images/26d142e321927ee23274b644e4a1459e03c0ee98d4d7d3178da0a9d18c0318e0.jpg)  
Ground Truth   
PSNR↑ / MAE↓

![](images/d62de6c5c101aec1443a7e1488d175bea5b8c92ca7833aaa15249ae9d171e358.jpg)  
Ours   
30.75 / 2.622

![](images/b8389995e9c7352b4505627cfd0fa03b4e3a2d723b0445d776aa22f7b4c1ff26.jpg)  
GS-IR   
25.47 / 7.252

![](images/652cfe85baf805e453b1c2d27e8d70f90aa38bdb9d960d479dce976bb81e5e77.jpg)  
GShader   
25.84 / 5.347

![](images/e23baef6d318873224a038e0b458a9f961f2f60374367ab47b6fd68e337c02db.jpg)  
Materials

![](images/59d051d63f5d4c77bfcc4b626362e4d3aee3a1c3e176ef386508b2ff18c7d77a.jpg)  
Indirect Light & Visibility

![](images/881c3159cd5d9d280416294eea695de67c283fecfd7162bba019ccfb47c281d1.jpg)  
Environment Map

![](images/190969227e500cd5928492fa44b8314dcd5dc31d4a70bd9c6c59cf29b28a674a.jpg)  
Relighting

![](images/43f926c9b08cefb2db3e9cd914091df72b92e47306a18bfe1f5f73164e9f5315.jpg)

![](images/90b174c37f61a7018f736d8e3c609926a7cb00addc1ab551889ab516a39f497a.jpg)  
Editing   
Figure 1: We propose RTR-GS, a framework for geometry-light-material decomposition from multi-view images. Our method significantly enhances normal estimation and visual realism for reflective surfaces compared to GS-IR [32] and GShader [23]. Additionally, we achieve material and lighting decomposition while accounting for secondary lighting effects through physically-based deferred rendering. The material components include albedo, metallic, and roughness. This high-quality decomposition enables realistic relighting and material editing.

# Abstract

3D Gaussian Splatting (3DGS) has demonstrated impressive capabilities in novel view synthesis. However, rendering reflective objects remains a significant challenge, particularly in inverse rendering and relighting. We introduce RTR-GS, a novel inverse rendering framework capable of robustly rendering objects with arbitrary

Corresponding author: Lei Zhang.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

MM ’25, Dublin, Ireland

$\circledcirc$ 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.

ACM ISBN 979-8-4007-2035-2/2025/10

https://doi.org/10.1145/3746027.3755197

reflectance properties, decomposing BRDF and lighting, and delivering credible relighting results. Given a collection of multi-view images, our method effectively recovers geometric structure through a hybrid rendering model that combines forward rendering for radiance transfer with deferred rendering for reflections. This approach successfully separates high-frequency and low-frequency appearances, mitigating floating artifacts caused by spherical harmonic overfitting when handling high-frequency details. We further refine BRDF and lighting decomposition using an additional physicallybased deferred rendering branch. Experimental results show that our method enhances novel view synthesis, normal estimation, decomposition, and relighting while maintaining efficient training inference process.

# CCS Concepts

• Computing methodologies Rasterization; Point-based models; Rendering; Machine learning.

# Keywords

Novel view synthesis, Gaussian Splatting, Relighting

# ACM Reference Format:

Yongyang Zhou, Fanglue Zhang, Zichen Wang, and Lei Zhang. 2025. RTR-GS: 3D Gaussian Splatting for Inverse Rendering with Radiance Transfer and Reflection. In Proceedings of the 33rd ACM International Conference on Multimedia (MM ’25), October 27–31, 2025, Dublin, Ireland. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3746027.3755197

# 1 Introduction

Inverse rendering is a long-standing challenge that seeks to decompose a 3D scene’s physical attributes—geometry, materials, and lighting—from captured images. This decomposition enables downstream tasks such as relighting and editing. The problem is particularly challenging due to the complex interplay of these attributes during rendering, especially under unknown illumination conditions, which make it inherently under-constrained. Neural Radiance Fields (NeRF) [37] have achieved remarkable success in novel view synthesis, laying the groundwork for inverse rendering. Methods such as [8, 33, 65, 67] use implicit neural representations, like Multi-Layer Perceptrons (MLPs), to decompose physical components. However, MLPs suffer from limited expressiveness and high computational costs, making it challenging to balance quality and efficiency. 3D Gaussian Splatting (3DGS) [26] improves both the speed and quality of learning-based volumetric rendering, and several methods [17, 32, 44] have integrated physically-based rendering into this framework. However, spherical harmonic functions lack the directional resolution needed to accurately represent specular reflections, and overfitting during Gaussian splatting and cloning can introduce floating artifacts.

Accurate geometry is crucial for decomposing materials and lighting from complex appearances. However, high-frequency details can cause overfitting, leading to floating artifacts that deviate from physically smooth surfaces and compromise geometric accuracy. To address this issue, we propose using a reflection map to store specular components, isolating high-frequency appearance details from the radiance component to mitigate overfitting. Additionally, we replace independent spherical harmonics with radiance transfer rendering, which imposes stronger global lowfrequency constraints when computing radiance components. By separating high-frequency and low-frequency appearances, our method enables accurate recovery of geometric structures with arbitrary reflectance properties. Following geometry reconstruction, we model occlusion and indirect illumination by baking visibility into 3D voxels and introducing indirect lighting parameters. This approach reduces aliasing artifacts in albedo, shadows, and lighting during decomposition. Finally, we achieve effective material and lighting decomposition by integrating an additional differentiable, physically-based deferred rendering branch.

The primary contribution of our work is the introduction of a Gaussian splatting-based inverse rendering framework, RTR-GS, which accurately estimates surface normals, bidirectional reflectance distribution functions (BRDF), and environmental lighting from multi-view images of both diffuse and specular objects. Specifically, it includes the following key aspects:

• We propose a 3DGS-based hybrid rendering model that integrates reflection maps with radiance transfer, effectively separating high-frequency and low-frequency appearances. This enables efficient rendering of objects with arbitrary reflectance properties while reducing floating artifacts.   
• We further enhance appearance decomposition through a dual-branch rendering approach, enabling efficient and accurate material and lighting decomposition via rational lighting modeling and occlusion data baked into 3D voxels.   
• Comprehensive experiments demonstrate that our method achieves state-of-the-art performance in novel view synthesis and relighting, producing credible results for both diffuse and specular objects.

# 2 RELATED WORK

# 2.1 Neural representations

Recently, NeRF [37] has garnered significant attention. Subsequent research has focused on enhancing rendering quality [3, 5, 27], improving surface reconstruction [30, 48, 57], and advancing object generation [12, 41, 50, 62], among other areas. Additionally, some methods aim to balance speed and quality [11, 13, 16, 21, 38, 46], facilitating more efficient evaluations.

3D Gaussian Splatting [26] effectively combines radiance field rendering with rasterization. Subsequent research has focused on enhancing rendering quality [34, 60], more accurate geometry reconstruction [22, 36, 61], expanding editability [35, 63], and increasing scalability [40]. However, these methods do not decompose appearance into materials and lighting, limiting their suitability for relighting and editing tasks.

# 2.2 Inverse rendering

Inverse rendering aims to decompose physically-based attributes from observations, including geometry, material, and lighting. A variety of methods simplify this problem by assuming controllable lighting conditions [1, 6, 7, 18, 42]. Some works relax these assumptions to consider direct lighting effects [2, 8, 9, 53, 65]. These works [14, 55, 56, 64, 67, 68] model secondary lighting effects. Some methods [24, 29] employ tensor decomposition techniques inspired by TensoRF [13]. NvDiffrec [39] and NvDiffrecMC [20] utilize differentiable rendering with rasterization or ray-tracing pipelines.

Methods based on 3D Gaussian Splatting (3DGS) have significantly accelerated training and rendering[52]. GS-IR [32], GIR [44], and R3DG [17] constrain surface normals using pseudo normals derived from depth and model shadows and indirect lighting through baking or ray-tracing. By leveraging pre-computed radiance transfer, PRT-GS [19] enables relighting, including secondary lighting effects. Phys3DGS [15] integrates 3D Gaussian splats with meshbased representations. Although these methods retain the high efficiency of 3DGS, using spherical harmonic functions as a radiance representation for geometry recovery often introduces floating artifacts on reflective surfaces, leading to geometric inaccuracies.

# 2.3 Reflective object reconstruction

Reconstructing reflective objects poses a significant challenge in inverse rendering tasks. Ref-NeRF [47] tries to address this by using

![](images/0d843635ae40176722feb7e2e4f361deccd4f0e17b83e2ed418ce6e1b37dd881.jpg)  
Figure 2: RTR-GS Rendering Pipeline. Our rendering pipeline consists of a hybrid rendering branch and a physically-based rendering branch. The hybrid rendering branch computes the radiance color for each Gaussian using forward rendering through radiance transfer, which is then blended with the reflections from deferred rendering after splatting. The physically-based rendering branch is fully implemented during the deferred rendering phase. Initially, the hybrid rendering branch reconstructs the fundamental geometric structure and stores visibility in voxel grids. The physically-based rendering branch is then activated to further decompose material appearances.

reflection directions instead of view directions. NeRO [33] explicitly models the reflection process. Deferred rendering approaches [31, 51, 58, 69] replace forward rendering to better handle reflections. GaussianShader [23] separates specular components and incorporates residual terms to capture secondary lighting effects. Additionally, PRD-GS [59] introduces progressive radiance distillation.

Inspired by these works, we adopt 3D Gaussians as the scene representation and develop an inverse rendering framework capable of effectively rendering object with arbitrary reflectance properties while also decomposing material and lighting components.

# 3 Method

# 3.1 Overview

Figure 2 illustrates the overall framework of the proposed RTR-GS. We initialize 3D Gaussians using sparse point clouds generated randomly or estimated by COLMAP [43]. To model reflections, it is essential to define the normals for the Gaussians. We define normals as the shortest axis of each Gaussian, oriented toward the viewing direction, and optimize them synergistically using deferred rendering of reflections and pseudo-normals derived from a depth map (Sec. 3.2). Subsequently, we refine the Gaussians by introducing additional parameters and integrating key components into a hybrid rendering model (Sec. 3.3). This model combines radiance from forward rendering with reflections from deferred rendering, effectively separating high-frequency and low-frequency appearances to better represent complex materials and achieve high-quality scene reconstruction. Next, we decompose the appearance using differentiable physically-based deferred rendering, incorporating occlusion

baking, indirect lighting modeling, and additional BRDF parameters. During this process, we employ two rendering branches simultaneously to refine the geometry (Sec. 3.4). Finally, we enhance the results through rendering losses and additional regularization terms (Sec. 3.5).

# 3.2 Deferred Rendering and Normal Modeling

In the 3DGS framework, the attributes of multiple Gaussians are blended in the image plane using splatting and alpha blending, as follows:

$$
I _ {f} = \sum_ {i = 0} ^ {N} f _ {i} \alpha_ {i} T _ {i} \tag {1}
$$

where $\alpha _ { i }$ is the opacity, $T _ { i } = \textstyle \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } )$ represents the accumulated transmittance, $f _ { i }$ denotes the parameters of the ??-th Gaussian, and $I _ { f }$ represents the splatted screen-space attribute buffer. In vanilla 3DGS [26], outgoing radiance is computed per-Gaussian before blending. This process is referred to as forward rendering. Additionally, the attributes associated with each Gaussian can be transformed into screen space for subsequent shading, a process known as deferred rendering. The following section explains our normal design and optimization based on the deferred rendering implementation.

Accurate normals are essential for modeling reflection. We define the normal direction as the shortest axis of the Gaussian. During the optimization process, the Gaussian shape typically flattens as it aligns with the surface, causing the shortest axis to correspond to a larger area. Similar to GS-IR [32] and R3DG [17], we optimize normals by enforcing consistency between the pseudo-normal map

$\hat { \mathbf { n } } _ { \mathbf { d } }$ , derived from the depth map, and the Gaussian normals map n, as follows:

$$
\mathcal {L} _ {n} = \| \mathbf {n} - \hat {\mathbf {n}} _ {\mathbf {d}} \| _ {2} \tag {2}
$$

This constraint is effective in optimizing normals when the depth map is smooth enough. Additionally, normals are used to compute reflection directions and contribute to deferred rendering. This process enables rendering losses to be backpropagated to the normals, refining the Gaussian shape. When specular reflection is dominant, rendering losses from reflections primarily drive normal optimization. Conversely, in diffuse regions, depth-derived pseudo-normals impose a stronger constraint. Figure 3 illustrates the normal optimization process. Inspired by 3DGS-DR [58], we also introduce a simplified normal propagation mechanism that periodically enhances Gaussian opacity, improving the model’s robustness against extreme specular reflections.

![](images/105df2eb532674597390e99e0a9bea059456d9297c16d15b4e7c90fb8c3efa91.jpg)  
Figure 3: By adjusting the shapes of the Gaussians using the pseudo normals and gradients from the reflection map, the normals are optimized.

# 3.3 Hybrid Rendering and Radiance Transfer

To effectively render appearances with diverse variations and to mitigate Gaussian floating artifacts caused by limited representation capability, we propose a hybrid rendering approach to replace the spherical harmonics-based forward rendering in 3DGS [26]. Our hybrid rendering model separates radiance and reflection to capture low-frequency and high-frequency components, respectively. Specifically, the radiance is computed using forward rendering, while the reflection is obtained through deferred rendering. The two components are then adaptively blended based on the reflection intensity as follows:

$$
I _ {r g b} = C _ {r} \cdot \left(1. 0 - R _ {i}\right) + C _ {r e f} \cdot R _ {i} \tag {3}
$$

where $C _ { r }$ is the radiance color, $C _ { r e f }$ is the reflection color, and $R _ { i }$ is the reflection intensity. The final blending is done in screen space. Further details on the reflection and radiance components are provided in the following sections.

Reflection. In forward rendering, BRDF lobes are computed individually using the respective normal of each Gaussian and are then blended after shading. However, this blending process broadens the final BRDF lobe, resulting in blurry rendering effects. In contrast, deferred rendering generates a single BRDF lobe based on the blended normal, providing higher precision and better preservation

of BRDF sharpness. Similar observations have been analyzed in GUS-IR [31] and GS-ROR [69].

For each Gaussian, we introduce additional reflection attributes for deferred rendering: reflection tint $R _ { t }$ and reflection roughness $R _ { r }$ . We adopt a microfacet BRDF to simulate surfaces with varying roughness levels and achieve efficient computation using the splitsum approximation [25]. The final reflection color is computed as:

$$
C _ {r e f} = R _ {t} \cdot F _ {r e f} \left(E _ {r}, R _ {r}, \mathbf {n}, \mathbf {v}\right) \tag {4}
$$

where $E _ { r }$ is a learnable reflection map, n and v denote the normal and the view direction, respectively. $F _ { r e f }$ represents the split-sum approximation [25], which will be explained in more detail in Section 3.4.

Radiance. Inspired by Precomputed Radiance Transfer (PRT) [45], we adopt radiance transfer instead of spherical harmonics to compute outgoing radiance. Firstly, we will describe how radiance transfer is used to shade each Gaussian, including both view-independent and view-dependent components. Then we will explain the motivation behind using radiance transfer.

The view-independent component is consistent with the radiance transfer rendering in PRT. This calculation approximates the diffuse part of rendering equation as a dot product of two vectors as follows:

$$
C _ {d} \approx \rho_ {d} \sum_ {j = 0} ^ {n ^ {2}} c _ {j} c _ {j} ^ {t} \tag {5}
$$

where $\pmb { \rho } _ { d }$ represents the diffuse base color, $c _ { j }$ denotes the coefficients of the spherical harmonics lighting, and $c _ { j } ^ { t }$ represents the transfer vector. Notably, all Gaussians share the same spherical harmonics lighting $c _ { j }$ but use individual transfer vector $c _ { j } ^ { t }$ .

For the view-dependent component, following the derivation in PRT [45], we need to compute a radiance transfer matrix to convert environmental lighting into transferred lighting. However, $n$ -order spherical harmonics lighting requires $n ^ { 2 }$ parameters to store the transfer matrix, leading to rapidly increasing storage costs as the number of Gaussians grows. To address this issue, we adopt neural radiance transfer for the view-dependent component and compute it in a manner similar to the view-independent case. Specifically, for each Gaussian, we introduce a set of randomly initialized radiance transfer features $f _ { t }$ and a specular base color $\rho _ { s }$ . We decode $f _ { t }$ and the reflection direction o using a lightweight MLP $G$ to obtain the neural radiance transfer vector $c _ { j } ^ { t } ( \mathbf { o } )$ . The view-dependent outgoing radiance is computed as:

$$
C _ {s} (\mathbf {o}) \approx \rho_ {s} \sum_ {j = 0} ^ {n ^ {2}} c _ {j} c _ {j} ^ {t} (\mathbf {o}), \quad \text {w i t h} \quad c _ {j} ^ {t} (\mathbf {o}) = G (f _ {t}, \mathbf {o}) \tag {6}
$$

The total outgoing radiance is given by $C _ { r } = C _ { d } \mathrm { { + } } C _ { s } ( \mathbf { o } )$ . After Gaussian splatting and blending, this radiance further participates in the blending process during deferred rendering. A detailed derivation of our radiance transfer implementation is provided in the supplementary materials.

Compared to spherical harmonics, radiance transfer allows us to maintain enougth representational capacity while providing stronger global low-frequency constraints. In the shading process,

all Gaussians share two global components: the spherical harmonics lighting $c _ { j }$ and the MLP ??. This design enables shading across Gaussians to be connected through shared components, promoting the representation of overall low-frequency variations. Meanwhile, each Gaussian has its own independent transfer vector and transfer features, along with base color attributes. This enables our radiance transfer representation to better handle components that are difficult to recover in the reflection part, such as local reflections and shadows. Figure 4 illustrates the differences between our radiance transfer representation and spherical harmonics in modeling the radiance component. While the rendering results exhibit comparable visual quality, radiance transfer demonstrates better performance in low-frequency component fitting, prevents artifact generation, and maintains geometric smoothness.

![](images/32b1842be8a756ecd5ca72fc4cee3f1c52602102f3830e3148d5ed2b1cae1b06.jpg)  
Figure 4: Radiance transfer provides a better representation of low-frequency appearances and helps prevent artifacts caused by overfitting high-frequency details. Such artifacts can degrade the smoothness of depth and normal estimations, reducing the quality of the reconstructed geometry.

# 3.4 Illumination Modeling and Decomposition

We primarily use differentiable physically-based deferred rendering to decompose appearance into material and lighting components. To prevent aliasing artifacts in shadows, lighting, and albedo, we leverage the recovered geometric structure to bake occlusion information into a voxel grid, following the approach in GS-IR [32]. Specifically, we set the background color to white and assign black to the Gaussian regions. The scene is then projected to generate a cubemap texture, which is converted into spherical harmonics coefficients and stored in the voxel grid.

For materials, we assign BRDF attributes to each Gaussian, including albedo c, metallic $m$ , and roughness ?? . For illumination, we use an environmental cubemap to implement image-based lighting (IBL) for handling direct lighting. Additionally, we add a parameter $L _ { i n d } \in [ 0 , 1 ] ^ { 3 }$ for each Gaussian to represent diffuse indirect lighting. The rendering equation $\begin{array} { r } { L ( \mathbf { o } ) = \int _ { \Omega } { L _ { i } ( \mathbf { i } ) f ( \mathbf { i } , \mathbf { o } ) ( \mathbf { i } \cdot \mathbf { n } ) d \mathbf { i } } } \end{array}$ is separated into diffuse and specular components to simplify computation. The diffuse component $L _ { d }$ is computed as follows:

$$
\begin{array}{l} L _ {d} (\mathbf {x}) = \frac {\mathbf {c}}{\pi} \int_ {\Omega} L _ {i} (\mathbf {x}, \mathbf {i}) (\mathbf {n} \cdot \mathbf {i}) d \mathbf {i} \\ = \frac {c}{\pi} \left[ \int_ {\Omega} L _ {i} ^ {d i r} (\mathbf {x}, \mathbf {i}) (\mathbf {n} \cdot \mathbf {i}) d \mathbf {i} + \int_ {\Omega} L _ {i} ^ {i n d} (\mathbf {x}, \mathbf {i}) (\mathbf {n}, \mathbf {i}) d \mathbf {i} \right] \tag {7} \\ \approx \frac {c}{\pi} \big [ V (\mathbf {x}) L _ {d} ^ {d i r} (\mathbf {x}) + (1 - V (\mathbf {x})) L _ {d} ^ {i n d} (\mathbf {x})) \big ] \\ \end{array}
$$

where $L _ { d } ^ { d i r } ( { \bf x } )$ represents the direct environmental illumination, which depends only on the normal direction n. This value is precomputed for efficiency and stored in a 2D texture. The indirect illumination $L _ { d } ^ { i n d } ( { \bf x } )$ is derived through the splatting and blending of $L _ { i n d }$ . The visibility term $V ( \mathbf { x } )$ is determined by applying trilinear interpolation to the precomputed spherical harmonics stored in the baked voxel grid.

For the specular $L _ { s }$ , we employ the split-sum approximation [25], treating it as the product of two independent integrals as follows:

$$
L _ {s} (\mathbf {x}, \mathbf {o}) \approx \int_ {\Omega} f _ {s} (\mathbf {i}, \mathbf {o}) (\mathbf {n} \cdot \mathbf {i}) d \mathrm {i} \int_ {\Omega} L _ {i} (\mathbf {x}, \mathbf {i}) D (\mathbf {i}, \mathbf {o}) (\mathbf {n} \cdot \mathbf {i}) d \mathrm {i} \tag {8}
$$

where $f ( \mathbf { i } , \mathbf { o } )$ represents the microfacet BRDF [10]. The first term of the integral represents the BRDF. It is precomputed and stored in a Look-Up Table (LUT). The second term accounts for the incoming radiance modulated by the normal distribution function (NDF) $D$ , which is pre-integrated and represented using a filtered cubemap. Finally, the outgoing radiance is expressed as:

$$
L _ {o} (\mathbf {x}, \mathbf {o}) = L _ {d} (\mathbf {x}) + L _ {s} (\mathbf {x}, \mathbf {o}) \tag {9}
$$

After completing deferred rendering, we obtain the PBR result $I _ { p b r }$

In the decomposition process, we use both the previously mentioned hybrid rendering and PBR branches simultaneously, rather than freezing the geometric parameters or enabling only the PBR branch. This approach is adopted for two main reasons. Firstly, different rendering models still require corresponding geometric adjustments for proper adaptation, so completely freezing the geometric parameters is undesirable. We need to locally optimize the geometric attributes of the Gaussian to accommodate the PBR branch. Secondly, since the PBR-related parameters are initialized randomly, using only PBR can easily lead to drastic changes in the geometric structure, which may render the baked visibility inapplicable. These two points will be further elaborated in the experimental section.

# 3.5 Optimization

Throughout the training process, we optimize the geometric attributes of the Gaussian, as well as various rendering attributes closely related to the two rendering branches, as illustrated by the 3D Gaussians in Figure 2. In addition, we need to optimize the small MLP $G$ , which is a 3-layer network with 64 hidden units, used to decode the transfer feature and reflection direction, as well as two $6 \times 1 2 8 \times 1 2 8$ cubemaps: the reflection map for hybrid rendering and the environment map for PBR. We first activate the hybrid rendering branch and optimize the corresponding parameters. After restoring the basic geometric structure, we then activate the PBR branch and optimize all parameters. Finally, we outline the primary loss function and the specialized regularization terms.

Rendering losses. As in 3DGS[26], we calculate the hybrid rendering loss $\mathcal { L } _ { H R }$ and PBR loss $\mathcal { L } _ { P B R }$ using the following equation:

$$
\mathcal {L} = (1 - \lambda) \mathcal {L} _ {1} (\hat {I}, I _ {g t}) + \lambda \mathcal {L} _ {D - S S I M} (\hat {I}, I _ {g t}) \tag {10}
$$

Light regularization. We apply a light regularization assuming a natural white incident light [33, 39] for optimizing environment map used in PBR as follows:

Table 1: NVS quality and FPS on TensoIR, Shiny Blender, Glossy Blender and Stanford ORB datasets. “HR” represents our hybrid rendering branch.   

<table><tr><td rowspan="2">Methods</td><td colspan="3">TensoIR</td><td colspan="3">Shiny Blender</td><td colspan="3">Stanford ORB</td><td colspan="3">Glossy Synthetic</td><td>FPS</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS</td><td></td></tr><tr><td>NeRO</td><td>32.60</td><td>0.933</td><td>0.082</td><td>30.96</td><td>0.953</td><td>0.081</td><td>29.25</td><td>0.970</td><td>0.060</td><td>29.73</td><td>0.950</td><td>0.057</td><td>&lt;1</td></tr><tr><td>TensoIR</td><td>35.18</td><td>0.976</td><td>0.040</td><td>27.95</td><td>0.896</td><td>0.159</td><td>34.81</td><td>0.983</td><td>0.029</td><td>25.88</td><td>0.904</td><td>0.112</td><td>4</td></tr><tr><td>GS-IR</td><td>34.80</td><td>0.960</td><td>0.047</td><td>26.98</td><td>0.874</td><td>0.152</td><td>32.95</td><td>0.928</td><td>0.054</td><td>23.48</td><td>0.813</td><td>0.167</td><td>189</td></tr><tr><td>R3DG</td><td>37.15</td><td>0.981</td><td>0.024</td><td>27.30</td><td>0.922</td><td>0.121</td><td>38.54</td><td>0.988</td><td>0.016</td><td>24.13</td><td>0.894</td><td>0.105</td><td>16</td></tr><tr><td>3DGS-DR</td><td>38.15</td><td>0.979</td><td>0.031</td><td>32.03</td><td>0.960</td><td>0.084</td><td>39.80</td><td>0.987</td><td>0.015</td><td>28.62</td><td>0.942</td><td>0.063</td><td>271</td></tr><tr><td>GShader</td><td>37.13</td><td>0.982</td><td>0.023</td><td>30.87</td><td>0.953</td><td>0.088</td><td>36.02</td><td>0.989</td><td>0.017</td><td>27.52</td><td>0.928</td><td>0.084</td><td>65</td></tr><tr><td>Ours</td><td>39.17</td><td>0.985</td><td>0.021</td><td>33.99</td><td>0.971</td><td>0.061</td><td>39.81</td><td>0.990</td><td>0.016</td><td>27.79</td><td>0.935</td><td>0.070</td><td>133</td></tr><tr><td>Ours(HR)</td><td>41.39</td><td>0.988</td><td>0.017</td><td>35.24</td><td>0.975</td><td>0.055</td><td>40.49</td><td>0.991</td><td>0.014</td><td>28.72</td><td>0.942</td><td>0.063</td><td>96</td></tr></table>

Table 2: Relighting quality is evaluated on the TensoIR, Shiny Blender, and Stanford ORB datasets.   

<table><tr><td rowspan="2">Methods</td><td colspan="3">TensoIR</td><td colspan="3">Shiny Blender</td><td colspan="3">Stanford ORB</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>TensoIR</td><td>28.55</td><td>0.945</td><td>0.080</td><td>22.30</td><td>0.842</td><td>0.184</td><td>26.22</td><td>0.947</td><td>0.049</td></tr><tr><td>GShader</td><td>26.86</td><td>0.930</td><td>0.063</td><td>19.20</td><td>0.874</td><td>0.131</td><td>26.23</td><td>0.952</td><td>0.043</td></tr><tr><td>GS-IR</td><td>25.98</td><td>0.897</td><td>0.092</td><td>21.18</td><td>0.846</td><td>0.160</td><td>28.44</td><td>0.960</td><td>0.038</td></tr><tr><td>R3DG</td><td>28.52</td><td>0.931</td><td>0.069</td><td>20.69</td><td>0.869</td><td>0.141</td><td>27.88</td><td>0.957</td><td>0.039</td></tr><tr><td>Ours</td><td>30.10</td><td>0.944</td><td>0.053</td><td>26.16</td><td>0.928</td><td>0.084</td><td>28.93</td><td>0.967</td><td>0.029</td></tr></table>

$$
\mathcal {L} _ {\text {l i g h t}} = \sum_ {c} \left(L _ {c} - \frac {1}{3} \sum_ {c} L _ {c}\right), c \in \{R, G, B \} \tag {11}
$$

Metal reflection prior. Due to the reflective properties of metals, we aim to make the metallic parameter $m$ in the PBR model as close as possible to the reflection intensity $R _ { i }$ in hybrid rendering, as follows:

$$
\mathcal {L} _ {m} = \mathcal {L} _ {1} (m, R _ {i}) \tag {12}
$$

which encourages our two rendering branches to maintain appearance consistency in high-frequency regions. The effectiveness of this regularization term is discussed in the following section. In addition, we incorporate a bilateral smoothness term $\mathcal { L } _ { s }$ and an object mask constraint $\mathcal { L } _ { o }$ . The final loss $\mathcal { L }$ is defined as:

$$
\mathcal {L} = \mathcal {L} _ {H R} + \lambda_ {P B R} \mathcal {L} _ {P B R} + \lambda_ {0} \mathcal {L} _ {\text {l i g h t}} + \lambda_ {1} \mathcal {L} _ {m} + \lambda_ {2} \mathcal {L} _ {n} + \mathcal {L} _ {s} + \mathcal {L} _ {o} \tag {13}
$$

where $\lambda _ { P B R } = 0$ or 1, $\lambda _ { 0 } ~ = ~ 0 . 0 0 3$ , $\lambda _ { 1 } = 0 . 1$ , $\lambda _ { 2 } = 0 . 0 2$ . Detailed descriptions of $\mathcal { L } _ { s }$ and $\mathcal { L } _ { o }$ are provided in the supplementary materials.

# 4 Experiments

# 4.1 Evaluation Setup

Dataset and Metrics. For synthetic objects in the TensoIR [24], Shiny Blender [47] and Glossy Blender [33] datasets, as well as real objects in the Stanford ORB dataset [28], we evaluate the performance of novel view synthesis using PSNR, SSIM [49], and LPIPS [66] metrics. In addition, we use mean angular error (MAE) to evaluate the quality of normal estimation. We have also provided the results of inference speed (FPS). We further evaluate novel view

synthesis on the Ref-Real [47] and MipNeRF-360 [4] datasets. Numbers in bold represent the best performance, while underscored numbers indicate the second-best performance. In addition, we perform relighting evaluation on three datasets[24, 28, 47].

Methods for Comparison. We compared the quality of novel view synthesis against several NeRF-based methods [24, 33] and 3DGS-based methods [17, 23, 32, 58]. In addition, we evaluated the relighting quality between different inverse rendering methods. All methods were implemented and trained using their publicly available code and default configurations.

# 4.2 Comparison with previous works

Novel view synthesis. Table 1 presents the quantitative comparison results for novel view synthesis (NVS) on object-level datasets. Our PBR results show clear advantages over other methods. Additionally, we provide our Hybrid Rendering (HR) branch results to demonstrate the effectiveness of the hybrid rendering model. Visual comparisons are provided in Figure 5. Notably, our method preserves stable geometric structures even with high-frequency surface variations, producing clearer and more accurate novel views. Furthermore, Table 3 presents our results on the Ref-Real dataset [47] and the Mip-NeRF 360 dataset [4], where our method achieves competitive quantitative results.

Relighting. Table 2 presents the results of the relighting comparison. For the TensoIR and Shiny Blender datasets, albedo is aligned to the ground truth via channel-wise scaling before relighting as described in [28, 54, 65]. For the Stanford ORB dataset, albedo scaling is disabled to more accurately evaluate absolute decomposition performance on real objects. Results for the TensoIR and Shiny Blender datasets are averaged over all viewpoints under five different environment maps. For the Stanford ORB dataset, relighting

![](images/23f133584f388f1177319d8d708edaebdc7d1a255bdcc95eea3123b35fa40ce3.jpg)  
GT

![](images/cf9f0fc562a12e9540a91215f6ce380a74b6ef103acb128c077c9023d9833950.jpg)  
Ours

![](images/9141c581c6316bbb7a7d6b9e4e69cf7062cd9448656bac0e3b9182ed06158d90.jpg)  
R3DG

![](images/8bdda36aa15f215a76c583e31a4a7d760c1fd043e1e64fe727c3bafacba39ed1.jpg)  
GS-IR

![](images/d42ad87940e576e5124e3f193bb3de8a68f14a8e10f733127aa94293d6dcd099.jpg)  
3DGS-DR

![](images/d61bd4f88a4dcd278317a3cc1e4830ee4e248dee0086e68fcd28e6134c43c830.jpg)  
GShader

![](images/3343de9cb077b4d4c4c0c9ffe5b37a4bb1a566dd966e8d64c13be18d278def2d.jpg)  
TensoIR

![](images/d4d4d12950c0f6774dbb65b18cf58fef4bb5cd5fdb8297d14661fc2eecc53f4e.jpg)

![](images/840ea9eba187816055fc985f87e4bd026aa02598ab362d8919105303af0433e9.jpg)

![](images/eb89a7771caa11e2b8577ffe2adaa1523f216f1ea30182ea7294fb8a7e46c83e.jpg)

![](images/a45a9175fb1f21d005e9c3fde05ff4fb9709db3e534b01264910df388804debd.jpg)

![](images/2d79ff2fa29caee867d2249c9acde5da54ee6b74d707625d9305b124d439d2b7.jpg)

![](images/cb0f71792d6ba52db6cc01b0fe726583359b3b9bb6c0d44ed5e8f505f3c05437.jpg)

![](images/99c332165147aaf96befd5ae828419a42c2b9fe78105c7f35127eff310bfe1ed.jpg)

![](images/82437bd0664067291c84d202e540a2faa99a28800c92f1607007708fbf891a3c.jpg)

![](images/6229306c179873209c4df521ea61e868bf9610e22960e63f4520d4860ff1bb56.jpg)

![](images/8c9c6ff510648512f648386d68f3c8744b53557a1bce818fa38c0bcf41cd02d6.jpg)

![](images/3b97d937f7caa0125f3e1a80371cab893e199ac77241bb4f3484b9159b610fef.jpg)

![](images/5f56e8d70f719fd97cbf5f73e20c33fc8c00ceb4f4fe71add4bdc632184686ad.jpg)

![](images/da1419a0c23cb1a5c199f4c2d03a7d552eb4f2c7078fad5408d6d9481e86a267.jpg)

![](images/612245c6698e69cff023c612f7c4a630a06d9a074e6a0cd2ccbb8ec8f19e7f52.jpg)

![](images/b83a4ab51643d1b74a44fc5c0d18e3d7d60b03a4474f8bc32160f3534907ba38.jpg)

![](images/4007c0ecb5ce03c6221b7c2e05398065eeddb875723936c12847334aad3db0b0.jpg)

![](images/d147bae1af721924ad3a30b5195207537fbca2ced2754bca52be8cdc100c3a71.jpg)

![](images/c804b696a34bc2809ed70fbc6b9e9f25c40da838d289e1de25febb2eb296aa19.jpg)

![](images/59b2a67ca7e08b5fc10d5e1c46dc9aaeeac231c1f4d2ffc895bba977376a344f.jpg)

![](images/881fc680a4d0337472b2d9b8d69072f716da718bd887fe25bb1f1d0e5cbe1d08.jpg)

![](images/627d3d32e6e5756d20e3d90534a567e812102ed41a8e024f6632c3a0af833a64.jpg)

![](images/1e2f1a32b312d9091a2bbbb98805252565386122cf0cb46b38c68c15257911f3.jpg)  
Figure 5: Qualitative comparisons on a synthetic dataset. Our method retains more details, particularly in specular regions.   
Figure 6: Qualitative comparisons of relighting with different environment lighting conditions.

![](images/1c09deb3ae220831213b1979d5349d6551f566988c8f33d0f597d415e26e2274.jpg)  
GT

![](images/cbc8dc12f09a62661dea4062833be61d99b7f4627d21723fba611e4a64abf0a5.jpg)  
Ours

![](images/e2fd2a2c1a7b25869157e74fd41a2597cc13e22ad0251a19fb3ded7b2c0b11df.jpg)  
R3DG

![](images/ebee9999db1be1f58199a8bf67eb0e28d14bf77b503c06e3d7c2db51b50569e2.jpg)  
GS-IR

![](images/b382997fa7df217986c35215e521d042985be40d6ee623f38204c9ae91cbca3f.jpg)  
3DGS-DR

![](images/1615522692eab84c56599c593691bbed37058527365762d9be4361d0ec7ad3f8.jpg)  
GShader

![](images/8850aae223b32979c4a89f4e2c0f33855d4abe1bf5bd286fd626e72219a83da6.jpg)  
TensoIR

![](images/2d91a47c3b4674584b20578c13812f5c2d9a321e5ed60bdc56d32d01c0dcb69e.jpg)

![](images/4bd556eed3033d83b8f7801e9209b83faceae86969e564505e042f589300beb6.jpg)

![](images/0f1216e07275e6123ec9e725825eab6f959e2561457ddce2fc77c7213529b60f.jpg)

![](images/22cfbb0962055bb6d3b62863c002d1dcb3882689337aae810d34ab9ca756473b.jpg)

![](images/106c056e724f2ccdacab95bd450b81c1e57bd05213ddca1d1389dd3a19ff50e1.jpg)

![](images/36d060511fb78ec0b1e9ec06c4a86995d411eaee32dce1b156935e781001c1b0.jpg)

![](images/e45282d4ce4f1f9cd5790b8f8103515904ecf1e98fa293a9dd67cee493070c75.jpg)  
Figure 7: Qualitative comparisons of normal estimated by different methods. Our method provides robust normal estimation.

is evaluated using the provided 20 image-environment map pairs. Visual comparisons are provided in Figure 6. Our method’s superior detail preservation and effectively suppresses aliasing artifacts in both albedo and lighting. Notably, our approach maintains credibility under different relighting conditions, without significant surface artifacts appearing on either rough or smooth objects.

Normal and materials estimation. Table 4 and Figure 7 present the results of our normal estimation. Notably, in the presence of high-frequency surface details, our method effectively prevents

surface discontinuities caused by floating artifacts. In Figure 9, we visualize the estimated albedo, metallic, roughness, normal, and environmental lighting components. Our framework successfully decomposes both diffuse and specular objects. For specular objects, we achieve high-quality decomposition results with clearer environmental lighting. Additional albedo estimation results and more qualitative comparisons are provided in the supplementary materials.

Table 3: Novel view synthesis quality evaluated using PSNR, SSIM, and LPIPS on the Ref-Real dataset and the Mip-NeRF 360 dataset.   

<table><tr><td rowspan="2">Methods</td><td colspan="3">Ref-Real</td><td colspan="3">Mip-NeRF 360</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>GS-IR</td><td>23.41</td><td>0.606</td><td>0.297</td><td>26.18</td><td>0.801</td><td>0.200</td></tr><tr><td>GShader</td><td>21.13</td><td>0.578</td><td>0.375</td><td>22.33</td><td>0.577</td><td>0.329</td></tr><tr><td>3DGS-DR</td><td>23.51</td><td>0.638</td><td>0.343</td><td>25.14</td><td>0.783</td><td>0.304</td></tr><tr><td>Ours</td><td>23.54</td><td>0.627</td><td>0.337</td><td>26.65</td><td>0.806</td><td>0.233</td></tr></table>

Table 4: Normal estimation quality with Gaussian-based methods evaluated using MAE↓ on the TensoIR dataset and the Shiny Blender dataset.   

<table><tr><td></td><td>GS-IR</td><td>R3DG</td><td>3DGS-DR</td><td>GShader</td><td>Ours</td></tr><tr><td>TensoIR</td><td>5.313</td><td>5.914</td><td>5.728</td><td>5.303</td><td>5.347</td></tr><tr><td>Shiny Blender</td><td>9.328</td><td>9.238</td><td>3.632</td><td>4.800</td><td>3.091</td></tr></table>

Table 5: Ablation study of key components on the Shiny Blender dataset. "w/o radiance transfer" represents using SHs to calculate the radiance part in hybrid rendering. "Propagation" denotes simplified normal propagation.   

<table><tr><td>Ablations</td><td>NVS PSNR↑</td><td>Relighting PSNR↑</td></tr><tr><td>ours</td><td>33.99</td><td>26.16</td></tr><tr><td>w/o radiance transfer</td><td>32.15</td><td>25.85</td></tr><tr><td>w/o propagation</td><td>33.26</td><td>26.09</td></tr><tr><td>w/o Lm</td><td>33.76</td><td>25.88</td></tr><tr><td>w/ frozen geometry</td><td>31.49</td><td>24.66</td></tr><tr><td>w/o hybrid rendering</td><td>32.90</td><td>25.18</td></tr></table>

![](images/b64efb13ac12d64999ca927827e497c31ccea94961e9170d1540a88b5ea46fb8.jpg)  
Figure 8: Radiance transfer can more effectively separate lowfrequency components of appearance, thereby preventing artifacts caused by overfitting. These artifacts compromise geometric smoothness and degrade the quality of rendering and relighting.

# 4.3 Ablation Study

We specifically evaluated the effectiveness of radiance transfer compared to SHs. Additionally, we performed ablation studies on

![](images/ecbd5db110027ea964745a5d5dea517cb0b014727fd45d2fbf01ee2ba1db21d7.jpg)  
Rendering

![](images/c22de8934f9181e6a38531ad2b047a4ccf841dd0249097b365a93bb526aeeb5c.jpg)

![](images/eb2411b0e8e14d22761013b406f5ae89b8e4cfe29bad2ab4293ccb8adcf93c13.jpg)  
Albedo Metallic Roughness Normal Environment Lighting   
Figure 9: Normal, albedo, roughness, metallic and environment lighing results on synthetic dataset.

simplified normal propagation to validate the contribution of our proposed components. We also evaluate the impact of the metal reflection prior introduced in Sec. 3.5. For decomposition process, we further conducted experiments of using fixed geometric parameters and disabling the hybrid rendering branch (i.e., using only the PBR branch) during decomposition, to demonstrate the advantages of our dual-branch rendering framework.

Analysis on radiance transfer. As illustrated in Figure 8, using radiance transfer instead of SHs to represent the radiance component in hybrid rendering reduces floating artifacts and prevents normal and visibility errors caused by local geometric inaccuracies, particularly for specular objects. These improvements significantly enhance the quality of relighting. As shown in Table 5, radiance transfer also leads to notable improvements in quantitative results.

Analysis on decomposition process. When decomposing the appearance, we simultaneously enable hybrid rendering and PBR to fine-tune the geometry, making it compatible with both rendering models. We also evaluate the effects of freezing geometric parameters or enabling only the PBR branch, which demonstrates the limitations of single-branch approaches. As shown in Table 5, both frozen geometry and enabling the PBR branch only lead to significant quality degradation. The former occurs because the geometric structure required for hybrid rendering does not fully meet PBR’s requirements, while the latter leads to geometric mutations, rendering the baked occlusion ineffective.

Limitation We assume that lighting originates from an infinite distance, which differs from actual lighting conditions in large-scale scenes. Additionally, our method does not consider more complex indirect lighting effects, such as inter-reflections.

# 5 Conclusions

We introduce RTR-GS, an inverse rendering framework that enables realistic novel view synthesis and relighting through Gaussian splatting and deferred rendering. By separating high-frequency and lowfrequency appearances using reflection maps and radiance transfer, we achieve high-quality hybrid rendering and normal estimation. Building on this, we further decompose material and lighting from the appearance by an additional PBR branch. Experimental results demonstrate that our method delivers competitive performance in novel view synthesis and relighting across various objects. In the future, we aim to explore more precise rendering techniques and incorporate more complex secondary lighting effects.

# Acknowledgments

This work was supported in part by the National Natural Science Foundation of China (62132012) and Natural Science Foundation of Shandong Province (Major Basic Research) project (ZR2024ZD12). Fanglue Zhang was supported by the Faculty Strategic Research Grant FSRG-ENGRADI-12684 from Victoria University of Wellington.

# References

[1] Dejan Azinovic, Tzu-Mao Li, Anton Kaplanyan, and Matthias Nießner. 2019. Inverse path tracing for joint material and lighting estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2447–2456.   
[2] Zhongyun Bao, Gang Fu, Zipei Chen, and Chunxia Xiao. 2024. Illuminator: Image-based illumination editing for indoor scene harmonization. Computational Visual Media 10, 6 (2024), 1137–1155.   
[3] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. 2021. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 5855–5864.   
[4] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. 2022. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5470–5479.   
[5] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. 2023. Zip-nerf: Anti-aliased grid-based neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 19697– 19705.   
[6] Sai Bi, Zexiang Xu, Kalyan Sunkavalli, Miloš Hašan, Yannick Hold-Geoffroy, David Kriegman, and Ravi Ramamoorthi. 2020. Deep reflectance volumes: Relightable reconstructions from multi-view photometric images. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part III 16. Springer, 294–311.   
[7] Sai Bi, Zexiang Xu, Kalyan Sunkavalli, David Kriegman, and Ravi Ramamoorthi. 2020. Deep 3d capture: Geometry and reflectance from sparse multi-view images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5960–5969.   
[8] Mark Boss, Raphael Braun, Varun Jampani, Jonathan T Barron, Ce Liu, and Hendrik Lensch. 2021. Nerd: Neural reflectance decomposition from image collections. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 12684–12694.   
[9] Mark Boss, Varun Jampani, Raphael Braun, Ce Liu, Jonathan Barron, and Hendrik Lensch. 2021. Neural-pil: Neural pre-integrated lighting for reflectance decomposition. Advances in Neural Information Processing Systems 34 (2021), 10691–10704.   
[10] Brent Burley and Walt Disney Animation Studios. 2012. Physically-based shading at disney. In Acm Siggraph, Vol. 2012. vol. 2012, 1–7.   
[11] Ang Cao and Justin Johnson. 2023. Hexplane: A fast representation for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 130–141.   
[12] Eric R Chan, Marco Monteiro, Petr Kellnhofer, Jiajun Wu, and Gordon Wetzstein. 2021. pi-gan: Periodic implicit generative adversarial networks for 3d-aware image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5799–5809.   
[13] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. 2022. Tensorf: Tensorial radiance fields. In European Conference on Computer Vision. Springer, 333–350.   
[14] Hao Chen, Bo He, Hanyu Wang, Yixuan Ren, Ser Nam Lim, and Abhinav Shrivastava. 2021. Nerv: Neural representations for videos. Advances in Neural Information Processing Systems 34 (2021), 21557–21568.   
[15] Euntae Choi and Sungjoo Yoo. 2024. Phys3DGS: Physically-based 3D Gaussian splatting for inverse rendering. arXiv preprint arXiv:2409.10335 (2024).   
[16] Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk Warburg, Benjamin Recht, and Angjoo Kanazawa. 2023. K-planes: Explicit radiance fields in space, time, and appearance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 12479–12488.   
[17] Jian Gao, Chun Gu, Youtian Lin, Zhihao Li, Hao Zhu, Xun Cao, Li Zhang, and Yao Yao. 2025. Relightable 3D Gaussians: realistic point cloud relighting with BRDF decomposition and ray tracing. In European Conference on Computer Vision. Springer, 73–89.   
[18] Kaiwen Guo, Peter Lincoln, Philip Davidson, Jay Busch, Xueming Yu, Matt Whalen, Geoff Harvey, Sergio Orts-Escolano, Rohit Pandey, Jason Dourgarian, et al. 2019. The relightables: Volumetric performance capture of humans with realistic relighting. ACM Transactions on Graphics (ToG) 38, 6 (2019), 1–19.   
[19] Yijia Guo, Yuanxi Bai, Liwen Hu, Ziyi Guo, Mianzhi Liu, Yu Cai, Tiejun Huang, and Lei Ma. 2024. PRTGS: Precomputed radiance transfer of gaussian Splats for

real-time high-quality relighting. In Proceedings of the 32nd ACM International Conference on Multimedia. 5112–5120.   
[20] Jon Hasselgren, Nikolai Hofmann, and Jacob Munkberg. 2022. Shape, light, and material decomposition from images using monte carlo rendering and denoising. Advances in Neural Information Processing Systems 35 (2022), 22856–22869.   
[21] Peter Hedman, Pratul P Srinivasan, Ben Mildenhall, Jonathan T Barron, and Paul Debevec. 2021. Baking neural radiance fields for real-time view synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 5875– 5884.   
[22] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2024. 2d Gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 Conference Papers. 1–11.   
[23] Yingwenqi Jiang, Jiadong Tu, Yuan Liu, Xifeng Gao, Xiaoxiao Long, Wenping Wang, and Yuexin Ma. 2024. Gaussianshader: 3d gaussian splatting with shading functions for reflective surfaces. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5322–5332.   
[24] Haian Jin, Isabella Liu, Peijia Xu, Xiaoshuai Zhang, Songfang Han, Sai Bi, Xiaowei Zhou, Zexiang Xu, and Hao Su. 2023. Tensoir: Tensorial inverse rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 165–174.   
[25] Brian Karis and Epic Games. 2013. Real shading in unreal engine 4. Proc. Physically Based Shading Theory Practice 4, 3 (2013), 1.   
[26] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023. 3D Gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42, 4 (2023), 139–1.   
[27] Simin Kou, Fanglue Zhang, Jakob Nazarenus, Reinhard Koch, and Neil A Dodgson. 2025. OmniPlane: A recolorable representation for dynamic scenes in omnidirectional videos. IEEE Transactions on Visualization and Computer Graphics (2025).   
[28] Zhengfei Kuang, Yunzhi Zhang, Hongxing Yu, Samir Agarwala, Elliott Wu, Jiajun Wu, et al. 2023. Stanford-orb: a real-world 3d object inverse rendering benchmark. Advances in Neural Information Processing Systems 36 (2023), 46938–46957.   
[29] Jia Li, Lu Wang, Lei Zhang, and Beibei Wang. 2024. Tensosdf: Roughness-aware tensorial representation for robust geometry and material reconstruction. ACM Transactions on Graphics (TOG) 43, 4 (2024), 1–13.   
[30] Zhaoshuo Li, Thomas Müller, Alex Evans, Russell H Taylor, Mathias Unberath, Ming-Yu Liu, and ChenHsuan Lin. 2023. Neuralangelo: High-fidelity neural surface reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8456–8465.   
[31] Zhihao Liang, Hongdong Li, Kui Jia, Kailing Guo, and Qi Zhang. 2024. GUS-IR: Gaussian splatting with unified shading for inverse rendering. arXiv preprint arXiv:2411.07478 (2024).   
[32] Zhihao Liang, Qi Zhang, Ying Feng, Ying Shan, and Kui Jia. 2024. Gs-ir: 3d gaussian splatting for inverse rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 21644–21653.   
[33] Yuan Liu, Peng Wang, Cheng Lin, Xiaoxiao Long, Jiepeng Wang, Lingjie Liu, Taku Komura, and Wenping Wang. 2023. Nero: Neural geometry and brdf reconstruction of reflective objects from multiview images. ACM Transactions on Graphics (TOG) 42, 4 (2023), 1–22.   
[34] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. 2024. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 20654–20664.   
[35] Guan Luo, Tianxing Xu, Yingtian Liu, Xiaoxiong Fan, Fanglue Zhang, and Songhai Zhang. 2024. 3D Gaussian editing with a single image. In Proceedings of the 32nd ACM International Conference on Multimedia. 6627–6636.   
[36] Xiaoyang Lyu, YangTian Sun, YiHua Huang, Xiuzhe Wu, Ziyi Yang, Yilun Chen, Jiangmiao Pang, and Xiaojuan Qi. 2024. 3dgsr: Implicit surface reconstruction with 3d gaussian splatting. arXiv preprint arXiv:2404.00409 (2024).   
[37] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Commun. ACM 65, 1 (2021), 99–106.   
[38] Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. 2022. Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics (TOG) 41, 4 (2022), 1–15.   
[39] Jacob Munkberg, Jon Hasselgren, Tianchang Shen, Jun Gao, Wenzheng Chen, Alex Evans, Thomas Müller, and Sanja Fidler. 2022. Extracting triangular 3d models, materials, and lighting from images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8280–8290.   
[40] Jakob Nazarenus, Simin Kou, FangLue Zhang, and Reinhard Koch. 2024. Arbitrary optics for Gaussian splatting using space warping. Journal of Imaging 10, 12 (2024), 330.   
[41] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. 2022. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988 (2022).   
[42] Carolin Schmitt, Simon Donne, Gernot Riegler, Vladlen Koltun, and Andreas Geiger. 2020. On joint estimation of pose, geometry and svbrdf from a handheld scanner. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern

Recognition. 3493–3503.   
[43] Johannes L Schonberger and Jan-Michael Frahm. 2016. Structure-from-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition. 4104–4113.   
[44] Yahao Shi, Yanmin Wu, Chenming Wu, Xing Liu, Chen Zhao, Haocheng Feng, Jingtuo Liu, Liangjun Zhang, Jian Zhang, Bin Zhou, et al. 2023. Gir: 3d gaussian inverse rendering for relightable scene factorization. arXiv preprint arXiv:2312.05133 (2023).   
[45] Peter-Pike Sloan, Jan Kautz, and John Snyder. 2023. Precomputed radiance transfer for real-time rendering in dynamic, low-frequency lighting environments. In Seminal Graphics Papers: Pushing the Boundaries, Volume 2. 339–348.   
[46] Cheng Sun, Min Sun, and Hwann-Tzong Chen. 2022. Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5459–5469.   
[47] Dor Verbin, Peter Hedman, Ben Mildenhall, Todd Zickler, Jonathan T Barron, and Pratul P Srinivasan. 2022. Ref-nerf: Structured view-dependent appearance for neural radiance fields. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 5481–5490.   
[48] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. 2021. Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. arXiv preprint arXiv:2106.10689 (2021).   
[49] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. 2004. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing 13, 4 (2004), 600–612.   
[50] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. 2024. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in Neural Information Processing Systems 36 (2024).   
[51] Tong Wu, Jiamu Sun, Yukun Lai, Yuewen Ma, Leif Kobbelt, and Lin Gao. 2024. DeferredGS: Decoupled and editable Gaussian splatting with deferred shading. arXiv preprint arXiv:2404.09412 (2024).   
[52] Tong Wu, Yujie Yuan, Lingxiao Zhang, Jie Yang, Yanpei Cao, Lingqi Yan, and Lin Gao. 2024. Recent advances in 3d gaussian splatting. Computational Visual Media 10, 4 (2024), 613–642.   
[53] Youxin Xing, Gaole Pan, Xiang Chen, Ji Wu, Lu Wang, and Beibei Wang. 2024. Real-time all-frequency global illumination with radiance caching. Computational Visual Media 10, 5 (2024), 923–936.   
[54] Rongkai Xu, Lei Zhang, and Fanglue Zhang. 2024. Intrinsic omnidirectional image decomposition with illumination pre-extraction. IEEE Transactions on Visualization and Computer Graphics 30, 7 (2024), 4416–4428.   
[55] Ziyi Yang, Yanzhen Chen, Xinyu Gao, Yazhen Yuan, Yu Wu, Xiaowei Zhou, and Xiaogang Jin. 2023. Sire-ir: Inverse rendering for brdf reconstruction with shadow and illumination removal in high-illuminance scenes. arXiv preprint arXiv:2310.13030 (2023).   
[56] Yao Yao, Jingyang Zhang, Jingbo Liu, Yihang Qu, Tian Fang, David McKinnon, Yanghai Tsin, and Long Quan. 2022. Neilf: Neural incident light field for

physically-based material estimation. In European Conference on Computer Vision. Springer, 700–716.   
[57] Lior Yariv, Jiatao Gu, Yoni Kasten, and Yaron Lipman. 2021. Volume rendering of neural implicit surfaces. Advances in Neural Information Processing Systems 34 (2021), 4805–4815.   
[58] Keyang Ye, Qiming Hou, and Kun Zhou. 2024. 3d gaussian splatting with deferred reflection. In ACM SIGGRAPH 2024 Conference Papers. 1–10.   
[59] Keyang Ye, Qiming Hou, and Kun Zhou. 2024. Progressive radiance distillation for inverse rendering with Gaussian splatting. arXiv preprint arXiv:2408.07595 (2024).   
[60] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. 2024. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 19447–19456.   
[61] Zehao Yu, Torsten Sattler, and Andreas Geiger. 2024. Gaussian opacity fields: Efficient and compact surface reconstruction in unbounded scenes. arXiv preprint arXiv:2404.10772 (2024).   
[62] YuJie Yuan, Xinyang Han, Yue He, FangLue Zhang, and Lin Gao. 2024. Munerf: Robust makeup transfer in neural radiance fields. IEEE Transactions on Visualization and Computer Graphics (2024).   
[63] Dingxi Zhang, Yujie Yuan, Zhuoxun Chen, Fanglue Zhang, Zhenliang He, Shiguang Shan, and Lin Gao. 2024. Stylizedgs: Controllable stylization for 3d gaussian splatting. arXiv preprint arXiv:2404.05220 (2024).   
[64] Jingyang Zhang, Yao Yao, Shiwei Li, Jingbo Liu, Tian Fang, David McKinnon, Yanghai Tsin, and Long Quan. 2023. Neilf++: Inter-reflectable light fields for geometry and material estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 3601–3610.   
[65] Kai Zhang, Fujun Luan, Qianqian Wang, Kavita Bala, and Noah Snavely. 2021. Physg: Inverse rendering with spherical gaussians for physics-based material editing and relighting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5453–5462.   
[66] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 586–595.   
[67] Xiuming Zhang, Pratul P Srinivasan, Boyang Deng, Paul Debevec, William T Freeman, and Jonathan T Barron. 2021. Nerfactor: Neural factorization of shape and reflectance under an unknown illumination. ACM Transactions on Graphics (ToG) 40, 6 (2021), 1–18.   
[68] Yuanqing Zhang, Jiaming Sun, Xingyi He, Huan Fu, Rongfei Jia, and Xiaowei Zhou. 2022. Modeling indirect illumination for inverse rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 18643– 18652.   
[69] Zuoliang Zhu, Beibei Wang, and Jian Yang. 2024. Gs-ror: 3d gaussian splatting for reflective object relighting via sdf priors. arXiv preprint arXiv:2406.18544 (2024).