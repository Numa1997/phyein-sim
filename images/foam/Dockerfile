# phyein-foam: OpenFOAM v2512 (openfoam.com / ESI) + ParaView (pvbatch), Gmsh, ffmpeg, Python.
# Base pinned by digest (opencfd/openfoam-default:2512, Ubuntu 24.04, pulled 11 Nov 2026).
FROM opencfd/openfoam-default@sha256:33fb575aa9980d2bc42fd58c75ae698c489293ba30c991380fe3f899c622f319

ENV DEBIAN_FRONTEND=noninteractive
# python3 + scientific stack from the distribution; ParaView with its Python (pvpython, pvbatch);
# xvfb gives pvbatch a virtual display; ffmpeg turns frames into a film.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-numpy python3-scipy python3-matplotlib \
        python3-paraview xvfb xauth libgl1-mesa-dri ffmpeg git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Gmsh, VTK and PyVista from PyPI, exact versions, in their OWN virtual environment (/opt/venv).
# They must not share a Python with ParaView: pip's VTK 9.7 and ParaView's own VTK 9.1 in one interpreter crash pvbatch.
# python3 = system Python (numpy, scipy, matplotlib, ParaView); /opt/venv/bin/python = gmsh, vtk, pyvista.
COPY images/foam/requirements.txt /opt/phyein-sim/requirements-foam.txt
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -r /opt/phyein-sim/requirements-foam.txt

COPY tools /opt/phyein-sim/tools
COPY cases/foam /opt/phyein-sim/cases/foam
COPY images/foam/selftest.sh /opt/phyein-sim/selftest.sh
COPY images/foam/entry.sh /opt/phyein-sim/entry.sh
ENV PHYEIN_SIM=/opt/phyein-sim \
    PATH=/opt/phyein-sim/tools:${PATH} \
    LIBGL_ALWAYS_SOFTWARE=1 \
    MPLBACKEND=Agg
WORKDIR /work
# entry.sh sources OpenFOAM's environment, then runs the command; default is the self-test
ENTRYPOINT ["/opt/phyein-sim/entry.sh"]
CMD ["/opt/phyein-sim/selftest.sh"]
