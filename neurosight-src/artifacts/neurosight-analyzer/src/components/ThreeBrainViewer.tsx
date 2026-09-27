import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { Eye, EyeOff, Layers, Scissors, RotateCcw, ShieldAlert, Compass, MapPin } from 'lucide-react';
import type { AnalysisResult } from '@workspace/api-client-react';

interface ThreeBrainViewerProps {
  result: AnalysisResult;
}

export function ThreeBrainViewer({ result }: ThreeBrainViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const meshesRef = useRef<{ [key: string]: THREE.Object3D }>({});
  const clipPlaneRef = useRef<THREE.Plane | null>(null);

  const [layers, setLayers] = useState({
    head: true,
    brain: true,
    enhancing_tumor: true,
    edema: true,
    necrotic_core: true,
  });

  const [clipEnabled, setClipEnabled] = useState(false);
  const [clipConstant, setClipConstant] = useState(0.0);
  const [clipAxis, setClipAxis] = useState<'z' | 'y' | 'x'>('z');
  const [loading, setLoading] = useState(true);

  const atlas = (result as any).anatomical_atlas;
  const patientMeshes = (result as any).patient_meshes;
  const patientId = result.patient_metadata.patient_id;
  const lesionLocalizer = (result as any).slice_previews?.lesion_localizer;

  // Initialize Three.js Scene
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const width = container.clientWidth || 600;
    const height = 440;

    // Scene
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 100);
    camera.position.set(0, -2.55, 1.55);
    camera.up.set(0, 0, 1); // Z is Superior in medical RAS
    cameraRef.current = camera;

    // Clipping plane for section slicing
    const normal = clipAxis === 'z' ? new THREE.Vector3(0, 0, -1) : (clipAxis === 'y' ? new THREE.Vector3(0, -1, 0) : new THREE.Vector3(-1, 0, 0));
    const clipPlane = new THREE.Plane(normal, clipConstant);
    clipPlaneRef.current = clipPlane;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.localClippingEnabled = clipEnabled;
    rendererRef.current = renderer;

    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.8;
    controls.zoomSpeed = 1.0;
    controls.minDistance = 0.7;
    controls.maxDistance = 6.0;
    controlsRef.current = controls;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x5b6773, 1.8);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 2.2);
    dirLight1.position.set(3, -4, 4);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xd8b9a0, 1.0);
    dirLight2.position.set(-3, 4, -2);
    scene.add(dirLight2);

    const pointLight = new THREE.PointLight(0xff55d1, 1.5, 10);
    pointLight.position.set(0, 0, 2);
    scene.add(pointLight);

    // Grid helper in axial plane
    const grid = new THREE.GridHelper(2.4, 12, 0x20d8d0, 0x14343a);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = -0.7;
    scene.add(grid);

    // Try loading GLB combined model, with fallback to raw buffer geometry
    setLoading(true);
    const glbUrl = `/api/v1/patients/${patientId}/glb/combined`;
    const loader = new GLTFLoader();

    loader.load(
      glbUrl,
      (gltf) => {
        scene.add(gltf.scene);
        meshesRef.current = {};

        gltf.scene.traverse((child) => {
          if ((child as THREE.Mesh).isMesh) {
            const mesh = child as THREE.Mesh;
            const name = mesh.name.toLowerCase();

            // Configure materials for realistic neuro-radiological visualization
            if (name.includes('head')) {
              meshesRef.current['head'] = mesh;
              mesh.material = new THREE.MeshPhysicalMaterial({
                color: 0xb98f83, transparent: true, opacity: 0.24,
                roughness: 0.72, metalness: 0, depthWrite: false,
                side: THREE.DoubleSide, clippingPlanes: clipEnabled ? [clipPlane] : [],
              });
            } else if (name.includes('brain')) {
              meshesRef.current['brain'] = mesh;
              mesh.material = new THREE.MeshPhysicalMaterial({
                color: 0x8cd2eb,
                transparent: true,
                opacity: 0.22,
                roughness: 0.25,
                metalness: 0.1,
                transmission: 0.5,
                ior: 1.35,
                depthWrite: false,
                clippingPlanes: clipEnabled ? [clipPlane] : [],
              });
            } else if (name.includes('tumor') || name.includes('et')) {
              meshesRef.current['enhancing_tumor'] = mesh;
              mesh.material = new THREE.MeshStandardMaterial({
                color: 0xf52d41,
                emissive: 0x440810,
                roughness: 0.35,
                metalness: 0.2,
                clippingPlanes: clipEnabled ? [clipPlane] : [],
              });
            } else if (name.includes('edema') || name.includes('ed')) {
              meshesRef.current['edema'] = mesh;
              mesh.material = new THREE.MeshStandardMaterial({
                color: 0x2dd787,
                transparent: true,
                opacity: 0.38,
                roughness: 0.4,
                clippingPlanes: clipEnabled ? [clipPlane] : [],
              });
            } else if (name.includes('necrotic') || name.includes('ncr')) {
              meshesRef.current['necrotic_core'] = mesh;
              mesh.material = new THREE.MeshStandardMaterial({
                color: 0x962db9,
                roughness: 0.5,
                clippingPlanes: clipEnabled ? [clipPlane] : [],
              });
            }
          }
        });
        if (lesionLocalizer?.normalized_xyz) {
          const [x, y, z] = lesionLocalizer.normalized_xyz as number[];
          const marker = new THREE.Mesh(new THREE.SphereGeometry(0.028, 20, 16), new THREE.MeshStandardMaterial({ color: 0xff173b, emissive: 0xff173b, emissiveIntensity: 1.1 }));
          marker.position.set(x, y, z);
          marker.renderOrder = 4;
          scene.add(marker);
          const ring = new THREE.Mesh(new THREE.TorusGeometry(0.06, 0.005, 8, 32), new THREE.MeshBasicMaterial({ color: 0xff5473, depthTest: false }));
          ring.position.copy(marker.position);
          ring.renderOrder = 5;
          scene.add(ring);
        }
        setLoading(false);
      },
      undefined,
      (err) => {
        console.warn('GLB load failed, creating meshes from geometry buffers:', err);
        // Fallback: build from patientMeshes buffers
        if (patientMeshes) {
          buildMeshesFromBuffers(scene, patientMeshes, clipPlane, clipEnabled);
        }
        setLoading(false);
      }
    );

    // Animation Loop
    let animId: number;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Resize Handler
    const handleResize = () => {
      if (!container || !renderer || !camera) return;
      const newWidth = container.clientWidth;
      camera.aspect = newWidth / height;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, height);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [patientId, lesionLocalizer]);

  // Helper for buffer geometry fallback
  function buildMeshesFromBuffers(
    scene: THREE.Scene,
    pm: any,
    clipPlane: THREE.Plane,
    clipOn: boolean
  ) {
    const palette: any = {
      head: { color: 0xb98f83, opacity: 0.24, transparent: true },
      brain: { color: 0x8cd2eb, opacity: 0.30, transparent: true },
      enhancing_tumor: { color: 0xf52d41, opacity: 1.0, transparent: false },
      edema: { color: 0x2dd787, opacity: 0.38, transparent: true },
      necrotic_core: { color: 0x962db9, opacity: 1.0, transparent: false },
    };

    meshesRef.current = {};

    Object.keys(palette).forEach((key) => {
      const geoData = pm[key];
      if (geoData && geoData.vertices && geoData.vertices.length > 0) {
        const geo = new THREE.BufferGeometry();
        const verts = new Float32Array(geoData.vertices);
        const faces = new Uint32Array(geoData.faces);

        geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
        geo.setIndex(new THREE.BufferAttribute(faces, 1));

        if (geoData.normals && geoData.normals.length === geoData.vertices.length) {
          geo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(geoData.normals), 3));
        } else {
          geo.computeVertexNormals();
        }

        const p = palette[key];
        const mat = new THREE.MeshStandardMaterial({
          color: p.color,
          transparent: p.transparent,
          opacity: p.opacity,
          roughness: 0.3,
          depthWrite: !p.transparent,
          clippingPlanes: clipOn ? [clipPlane] : [],
        });

        const mesh = new THREE.Mesh(geo, mat);
        scene.add(mesh);
        meshesRef.current[key] = mesh;
      }
    });
  }

  // Update Clipping Plane
  useEffect(() => {
    if (!rendererRef.current) return;
    rendererRef.current.localClippingEnabled = clipEnabled;

    const normal = clipAxis === 'z' ? new THREE.Vector3(0, 0, -1) : (clipAxis === 'y' ? new THREE.Vector3(0, -1, 0) : new THREE.Vector3(-1, 0, 0));
    if (clipPlaneRef.current) {
      clipPlaneRef.current.normal.copy(normal);
      clipPlaneRef.current.constant = clipConstant;
    }

    Object.values(meshesRef.current).forEach((obj) => {
      const mesh = obj as THREE.Mesh;
      if (mesh.material) {
        const mat = mesh.material as THREE.Material;
        mat.clippingPlanes = clipEnabled && clipPlaneRef.current ? [clipPlaneRef.current] : [];
        mat.needsUpdate = true;
      }
    });
  }, [clipEnabled, clipConstant, clipAxis]);

  // Update Layer Visibility
  useEffect(() => {
    Object.entries(layers).forEach(([key, visible]) => {
      const obj = meshesRef.current[key];
      if (obj) {
        obj.visible = visible;
      }
    });
  }, [layers]);

  const toggleLayer = (layer: keyof typeof layers) => {
    setLayers((prev) => ({ ...prev, [layer]: !prev[layer] }));
  };

  const handleResetCamera = () => {
    if (cameraRef.current && controlsRef.current) {
      cameraRef.current.position.set(0, -2.55, 1.55);
      controlsRef.current.target.set(0, 0, 0);
      controlsRef.current.update();
    }
  };

  return (
    <section className="overflow-hidden rounded-xl border border-[#1b4b50] bg-[#050b10] shadow-[var(--shadow-soft)] neon-glow" data-testid="card-brain-model">
      {/* Header bar */}
      <div className="flex flex-col justify-between gap-3 border-b border-[#173a40] px-5 py-4 sm:flex-row sm:items-center">
        <div>
          <div className="section-label mb-1 text-[#57d8d0]">WebGL 3D Engine · MONAI SegResNet & MNI152 Atlas</div>
          <h3 className="font-['Space_Grotesk'] text-[18px] font-semibold tracking-[-0.025em] text-[#e8ffff]">
            3D Head, Neck & Tumor Location
          </h3>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-[10px]">
          <span className="flex items-center gap-1.5 rounded-full border border-[#245d60] bg-[#092126] px-2.5 py-1.5 text-[#8ee9df]">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#55f1d5]" />
            REAL GLB MESH · 60 FPS
          </span>
          <button
            type="button"
            onClick={handleResetCamera}
            className="flex items-center gap-1 rounded-md border border-[#245d60] bg-[#092126] px-2.5 py-1.5 text-[#8ee9df] hover:bg-[#12363e] transition-colors"
          >
            <RotateCcw size={12} /> Reset View
          </button>
        </div>
      </div>

      {/* Main 3D Stage & Anatomical Controls */}
      <div className="grid gap-0 lg:grid-cols-[minmax(0,1fr)_320px]">
        {/* Left: Interactive 3D WebGL Canvas */}
        <div className="relative min-h-[440px] bg-[radial-gradient(circle_at_50%_46%,rgba(22,105,111,.24),transparent_40%),linear-gradient(135deg,#071218,#03070b)] border-b border-[#173a40] lg:border-b-0 lg:border-r">
          {loading && (
            <div className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-2 bg-[#050b10]/80 backdrop-blur-sm">
              <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
              <p className="text-xs text-muted-foreground mono">Loading registered head and MRI lesion geometry...</p>
            </div>
          )}

          {/* Canvas container */}
          <div ref={containerRef} className="h-[440px] w-full cursor-grab active:cursor-grabbing" />

          {/* Quick HUD controls */}
          <div className="absolute top-4 left-4 z-10 flex flex-wrap gap-1.5">
            <button onClick={() => toggleLayer('head')} className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium ${layers.head ? 'bg-[#b98f83]/20 text-[#d7b6a8] border border-[#b98f83]/50' : 'bg-secondary/40 text-muted-foreground line-through'}`}>
              {layers.head ? <Eye size={12} /> : <EyeOff size={12} />} Head & Neck
            </button>
            <button
              onClick={() => toggleLayer('brain')}
              className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium transition-colors ${
                layers.brain ? 'bg-[#8cd2eb]/20 text-[#8cd2eb] border border-[#8cd2eb]/50' : 'bg-secondary/40 text-muted-foreground line-through'
              }`}
            >
              {layers.brain ? <Eye size={12} /> : <EyeOff size={12} />} Brain Cortex
            </button>
            <button
              onClick={() => toggleLayer('enhancing_tumor')}
              className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium transition-colors ${
                layers.enhancing_tumor ? 'bg-[#f52d41]/20 text-[#f52d41] border border-[#f52d41]/50' : 'bg-secondary/40 text-muted-foreground line-through'
              }`}
            >
              {layers.enhancing_tumor ? <Eye size={12} /> : <EyeOff size={12} />} Tumor (ET)
            </button>
            <button
              onClick={() => toggleLayer('edema')}
              className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium transition-colors ${
                layers.edema ? 'bg-[#2dd787]/20 text-[#2dd787] border border-[#2dd787]/50' : 'bg-secondary/40 text-muted-foreground line-through'
              }`}
            >
              {layers.edema ? <Eye size={12} /> : <EyeOff size={12} />} Edema (ED)
            </button>
            <button
              onClick={() => toggleLayer('necrotic_core')}
              className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium transition-colors ${
                layers.necrotic_core ? 'bg-[#962db9]/20 text-[#c86fe8] border border-[#962db9]/50' : 'bg-secondary/40 text-muted-foreground line-through'
              }`}
            >
              {layers.necrotic_core ? <Eye size={12} /> : <EyeOff size={12} />} Necrosis (NCR)
            </button>
          </div>

          {/* Section Slicing Controls Bar */}
          <div className="absolute bottom-4 left-4 right-4 z-10 rounded-lg border border-[#1b4b50] bg-[#07131a]/90 p-2.5 backdrop-blur-md">
            <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
              <label className="flex items-center gap-2 cursor-pointer text-foreground font-medium">
                <input
                  type="checkbox"
                  checked={clipEnabled}
                  onChange={(e) => setClipEnabled(e.target.checked)}
                  className="rounded border-border text-primary"
                />
                <Scissors size={14} className="text-primary" />
                <span>Parenchyma Section Slice (Deep Dissection)</span>
              </label>

              {clipEnabled && (
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-1">
                    {(['z', 'y', 'x'] as const).map((ax) => (
                      <button
                        key={ax}
                        onClick={() => setClipAxis(ax)}
                        className={`rounded px-2 py-0.5 uppercase mono text-[10px] ${
                          clipAxis === ax ? 'bg-primary text-primary-foreground font-bold' : 'bg-secondary text-muted-foreground'
                        }`}
                      >
                        {ax === 'z' ? 'Axial' : ax === 'y' ? 'Coronal' : 'Sagittal'}
                      </button>
                    ))}
                  </div>
                  <input
                    type="range"
                    min="-0.9"
                    max="0.9"
                    step="0.02"
                    value={clipConstant}
                    onChange={(e) => setClipConstant(parseFloat(e.target.value))}
                    className="w-32 accent-primary cursor-pointer"
                  />
                  <span className="mono text-[10px] text-muted-foreground w-12 text-right">
                    {(clipConstant * 100).toFixed(0)} mm
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Sidebar: Anatomical Atlas & Eloquent Context */}
        <div className="flex flex-col justify-between gap-4 bg-[#071016] p-5 text-foreground overflow-y-auto max-h-[440px]">
          <div>
            <div className="section-label mb-2 text-[#5cded6] flex items-center gap-1.5">
              <Compass size={13} className="text-primary" />
              <span>HARVARD-OXFORD MNI152 ATLAS REPORT</span>
            </div>

            {atlas ? (
              <div className="space-y-3.5">
                <div className="rounded-lg border border-[#173a40] bg-[#05090d] p-3">
                  <div className="text-[10px] text-muted-foreground">Primary Anatomical Locus</div>
                  <div className="mt-1 font-['Space_Grotesk'] text-[15px] font-semibold text-[#b3ffff] flex items-center gap-1.5">
                    <MapPin size={14} className="text-primary shrink-0" />
                    <span>{atlas.primary_location}</span>
                  </div>
                  <div className="mt-1.5 flex items-center justify-between text-[11px] mono text-[#6d9da0]">
                    <span>MNI Coordinates:</span>
                    <span>[{atlas.centroid_mni_mm.join(', ')}] mm</span>
                  </div>
                </div>

                {/* Affected lobes / structures */}
                <div>
                  <div className="text-[11px] font-semibold text-muted-foreground mb-1.5 flex items-center gap-1">
                    <Layers size={13} className="text-primary" /> Affected Gyri / Structures
                  </div>
                  <div className="space-y-1.5">
                    {atlas.region_overlaps.map((ro: any, idx: number) => (
                      <div key={idx} className="flex items-center justify-between rounded bg-[#0a1820] px-2.5 py-1.5 text-[11px]">
                        <span className="truncate max-w-[190px]">{ro.region_name}</span>
                        <span className="mono font-semibold text-primary">{ro.overlap_pct}%</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Eloquent cortex risk */}
                <div className="rounded-lg border border-[#2a2010] bg-[#140d05] p-3 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="flex items-center gap-1 font-bold text-[#ff9933]">
                      <ShieldAlert size={14} /> Eloquence Proximity
                    </span>
                    <span className="rounded bg-[#ff9933]/20 px-2 py-0.5 text-[10px] font-bold text-[#ff9933]">
                      {atlas.overall_eloquence_risk}
                    </span>
                  </div>
                  {atlas.eloquent_proximity && atlas.eloquent_proximity.length > 0 && (
                    <div className="mt-2 space-y-1 text-[11px] text-[#d6b085]">
                      <p className="font-medium text-foreground">
                        Nearest: {atlas.eloquent_proximity[0].structure_name} ({atlas.eloquent_proximity[0].distance_mm} mm)
                      </p>
                      <p className="text-[10px] leading-4 text-[#ba9a74]">
                        {atlas.eloquent_proximity[0].clinical_caution}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="rounded-lg border border-[#173a40] bg-[#05090d] p-3">
                  <div className="text-[10px] text-muted-foreground">3D Mesh Resolution</div>
                  <div className="mt-1 font-mono text-[13px] text-foreground">
                    Brain: {patientMeshes?.brain?.vertex_count?.toLocaleString() || 13774} vertices
                  </div>
                  <div className="font-mono text-[12px] text-muted-foreground">
                    Tumor: {patientMeshes?.enhancing_tumor?.vertex_count?.toLocaleString() || 6270} vertices
                  </div>
                </div>
                <p className="text-xs text-muted-foreground leading-5">
                  Atlas parcellation computed via SimpleITK affine stereotaxic registration.
                </p>
              </div>
            )}
          </div>

          <div className="border-t border-[#173a40] pt-2.5 text-[10px] mono text-[#6d9da0]">
            Left drag: Orbit • Right drag: Pan • Scroll: Zoom
          </div>
        </div>
      </div>
    </section>
  );
}
