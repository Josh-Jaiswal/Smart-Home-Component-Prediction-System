import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

const SpaceWaveBackground = () => {
    const mountRef = useRef(null);

    useEffect(() => {
        // Scene setup
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.z = 5;

        // Renderer setup
        const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        mountRef.current.appendChild(renderer.domElement);

        // Create stars
        const starsGeometry = new THREE.BufferGeometry();
        const starVertices = [];
        const starColors = [];
        
        for (let i = 0; i < 1000; i++) {
            const x = (Math.random() - 0.5) * 20;
            const y = (Math.random() - 0.5) * 20;
            const z = (Math.random() - 0.5) * 20;
            
            starVertices.push(x, y, z);
            
            // Random star colors (blue/purple tint)
            const r = Math.random() * 0.3 + 0.7; // Mostly white
            const g = Math.random() * 0.3 + 0.7;
            const b = Math.random() * 0.2 + 0.8; // More blue
            
            starColors.push(r, g, b);
        }
        
        starsGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starVertices, 3));
        starsGeometry.setAttribute('color', new THREE.Float32BufferAttribute(starColors, 3));
        
        const starsMaterial = new THREE.PointsMaterial({
            size: 0.1,
            vertexColors: true,
            transparent: true
        });
        
        const starField = new THREE.Points(starsGeometry, starsMaterial);
        scene.add(starField);

        // Create wave geometry
        const waveGeometry = new THREE.PlaneGeometry(15, 15, 50, 50);
        const waveMaterial = new THREE.MeshBasicMaterial({
            color: 0x6c63ff,
            wireframe: true,
            transparent: true,
            opacity: 0.3
        });
        
        const wave = new THREE.Mesh(waveGeometry, waveMaterial);
        wave.rotation.x = -Math.PI / 2; // Rotate to horizontal
        wave.position.y = -2; // Position below the center
        scene.add(wave);
        
        // Animation function
        const animate = () => {
            requestAnimationFrame(animate);
            
            // Rotate star field slowly
            starField.rotation.y += 0.0005;
            starField.rotation.x += 0.0002;
            
            // Animate wave vertices
            const time = Date.now() * 0.001;
            const positions = waveGeometry.attributes.position;
            
            for (let i = 0; i < positions.count; i++) {
                const x = positions.getX(i);
                const y = positions.getY(i);
                
                // Create wave effect
                const amplitude = 0.3;
                const frequency = 0.5;
                const z = amplitude * Math.sin(x * frequency + time) * Math.cos(y * frequency + time);
                
                positions.setZ(i, z);
            }
            
            positions.needsUpdate = true;
            
            renderer.render(scene, camera);
        };
        
        animate();

        // Handle window resize
        const handleResize = () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        };
        
        window.addEventListener('resize', handleResize);

        // Cleanup
        return () => {
            window.removeEventListener('resize', handleResize);
            mountRef.current?.removeChild(renderer.domElement);
            scene.remove(starField);
            scene.remove(wave);
            waveGeometry.dispose();
            waveMaterial.dispose();
            starsGeometry.dispose();
            starsMaterial.dispose();
            renderer.dispose();
        };
    }, []);

    return <div ref={mountRef} className="space-wave-background" />;
};

export default SpaceWaveBackground;