# 3D Model Generator

A web-based tool that converts 2D images into 3D models using the TRELLIS pipeline.

## Last Updated
- **Date**: 2025-02-12
- **Time**: 09:42:47 UTC
- **Author**: Krizzna69

## Overview
This application provides a user-friendly interface for generating 3D models from 2D images using Hugging Face's TRELLIS model. It features real-time progress tracking, GPU quota management, and interactive 3D model viewing capabilities.

## Features
- 🖼️ Image to 3D conversion
- 📊 Real-time progress tracking
- 🎥 Video preview generation
- 📦 GLB model export
- 🖥️ Interactive 3D model viewer
- ⚡ GPU quota management

## Technical Details
- **Framework**: Gradio
- **AI Model**: TRELLIS (via Hugging Face)
- **Output Formats**: 
  - Video preview
  - GLB (3D model)
  - Interactive viewer

## Requirements
```python
gradio
gradio_client