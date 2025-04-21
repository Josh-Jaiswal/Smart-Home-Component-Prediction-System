import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

const Background3D = () => {
    const mountRef = useRef(null);

    useEffect(() => {
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.z = 5;

        const renderer = new THREE.WebGLRenderer({ alpha: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        mountRef.current.appendChild(renderer.domElement);

        const particles = new THREE.BufferGeometry();
        const vertices = [];
        for (let i = 0; i < 500; i++) {
            vertices.push((Math.random() - 0.5) * 10);
            vertices.push((Math.random() - 0.5) * 10);
            vertices.push((Math.random() - 0.5) * 10);
        }
        particles.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));

        const material = new THREE.PointsMaterial({ color: 0xffffff, size: 0.05 });
        const particleSystem = new THREE.Points(particles, material);
        scene.add(particleSystem);

        const animate = () => {
            requestAnimationFrame(animate);
            particleSystem.rotation.y += 0.001;
            renderer.render(scene, camera);
        };
        animate();

        return () => {
            mountRef.current.removeChild(renderer.domElement);
        };
    }, []);

    return <div ref={mountRef} className="background-3d" />;
};

export default Background3D;