# phyein-fem: DOLFINx / FEniCSx 0.11 (weak forms in Python, PETSc) + Gmsh 4.15 + CalculiX 2.21 + ParaView + ffmpeg + PyVista.
# Base pinned by digest (dolfinx/dolfinx:v0.11.0, Ubuntu 24.04, pulled 11 Nov 2026).
# Elmer is not installed: there is no Ubuntu 24.04 package (only a PPA); FEniCSx covers the same physics with symbolic weak forms.
FROM dolfinx/dolfinx@sha256:2ae4bfbc0d9077268880faf04c72750528bee986c94ab223a2c159969bd56fa8

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        calculix-ccx python3-paraview xvfb xauth libgl1-mesa-dri ffmpeg git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY images/fem/requirements.txt /opt/phyein-sim/requirements-fem.txt
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    pip install --no-cache-dir --break-system-packages -r /opt/phyein-sim/requirements-fem.txt

COPY tools /opt/phyein-sim/tools
COPY cases/fem /opt/phyein-sim/cases/fem
COPY images/fem/selftest.sh /opt/phyein-sim/selftest.sh
ENV PHYEIN_SIM=/opt/phyein-sim \
    PATH=/opt/phyein-sim/tools:${PATH} \
    LIBGL_ALWAYS_SOFTWARE=1 \
    MPLBACKEND=Agg \
    OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
WORKDIR /work
ENTRYPOINT []
CMD ["/opt/phyein-sim/selftest.sh"]
