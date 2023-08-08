#include<vector>
#include<cstdio>

#include<torch/extension.h>

#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include<pybind11/numpy.h>

#include"Fractal2DRenderer_cpu.h"
#include"IFSMap.h"

template<typename T>
using ndarray = pybind11::array_t<T>;

ndarray<double> generate(int npts, const std::vector<std::vector<std::vector<double>>> mapss, uint32_t pointgen_seed, int nfrac) {
	const int nmapss = mapss.size();
	ndarray<double> pts({nmapss,npts,2});
	double *pts_ = (double*)pts.data();
	std::vector<std::vector<IFSMap>> mapss_(nmapss);
// #ifdef _OPENMP
#pragma omp parallel for
// #endif
	for (int i = 0; i < nmapss; ++i) {
		mapss_[i].resize(mapss[i].size());
		for (int j = 0; j < mapss_[i].size(); ++j) {
			mapss_[i][j] = IFSMap{ mapss[i][j][0], mapss[i][j][1], mapss[i][j][2], mapss[i][j][3], mapss[i][j][4], mapss[i][j][5], mapss[i][j][6] };
		}
	}
	generatePoints_cpu(pts_, npts, mapss_, pointgen_seed, nfrac);
	return pts;
}

torch::Tensor render(const ndarray<double> pts, int width, int height, int patch_mode, int flip_flg, uint32_t patchgen_seed) {
	const int nbatch = pts.shape()[0];
	const int npts = pts.shape()[1];
	const double *pts_ = pts.data();
	torch::Tensor imgs = torch::zeros({ nbatch, width, height, 3 }, at::TensorOptions(at::kByte));
	unsigned char *imgs_ = imgs.data_ptr<unsigned char>();
	renderPoints_cpu(pts_, npts, nbatch, imgs_, width, height, patch_mode, flip_flg, patchgen_seed);
	return imgs;
}



PYBIND11_MODULE(PyFractal2DRenderer, m) {
	m.doc() = "PyFractal2DRenderer";
	m.def("generate", &generate, "generate points", py::arg("npts"), py::arg("mapss"), py::arg("pointgen_seed"),py::arg("nfrac"));
	m.def("render", &render, "render points", py::arg("pts"), py::arg("width"), py::arg("height"), py::arg("patch_mode"), py::arg("flip_flg"), py::arg("patchgen_seed"));
}

