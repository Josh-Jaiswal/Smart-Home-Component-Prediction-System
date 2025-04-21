import React, { useEffect } from 'react';
import './AnimatedBackground.css';

const AnimatedBackground = () => {
  useEffect(() => {
    document.body.classList.add('animated-bg');
    return () => document.body.classList.remove('animated-bg');
  }, []);

  return null;
};

export default AnimatedBackground;