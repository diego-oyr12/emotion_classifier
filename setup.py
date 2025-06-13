from setuptools import setup, find_packages

setup(
    name="pet-emotion-classifier",
    version="1.0.0",
    author="Tu Nombre",
    author_email="tu.email@ejemplo.com",
    description="Aplicación para clasificación de emociones en mascotas",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.1",
        "tensorflow>=2.13.0",
        "numpy>=1.24.3",
        "Pillow>=10.0.1",
        "opencv-python>=4.8.1.78",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "pet-emotion-classifier=app:main",
        ],
    },
)
