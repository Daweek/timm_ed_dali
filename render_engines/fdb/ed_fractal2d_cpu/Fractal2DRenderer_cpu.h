#pragma once
#include"IFSMap.h"
#include<vector>
#include <cstdint>
void generatePoints_cpu(double *pts, int npts, const std::vector<std::vector<IFSMap>> &mapss, uint64_t pointgen_seed, int nfrac);
void renderPoints_cpu(const double *pts, int npts, int ninstances, unsigned char *imgs, int width, int height, int patch_mode, int flip_flg, uint64_t patchgen_seed);
