#include "Fractal2DRenderer_cpu.h"
#include "IFSMap.h"

#include<vector>
#include<random>
#include<utility>
#ifdef OpenCV_FOUND
#include<opencv2/opencv.hpp>
#endif
#ifdef flann_FOUND
#include<flann/flann.hpp>
#endif

#include<chrono>
#include<cstdio>

#include <iostream>
#include <iomanip>
#include <random>
// #include <ippcore.h>
// #include<immintrin.h>


std::vector<float> hsv2rgb(const std::vector<float> &hsv) {
	float h = hsv[0];
	float s = hsv[1];
	float v = hsv[2];
	float r = v;
	float g = v;
	float b = v;
	if (s > 0.0f) {
		h *= 6.0f;
		const int i = (int)h;
		const float f = h - (float)i;
		switch (i) {
		default:
		case 0:
			g *= 1 - s * (1 - f);
			b *= 1 - s;
			break;
		case 1:
			r *= 1 - s * f;
			b *= 1 - s;
			break;
		case 2:
			r *= 1 - s;
			b *= 1 - s * (1 - f);
			break;
		case 3:
			r *= 1 - s;
			g *= 1 - s * f;
			break;
		case 4:
			r *= 1 - s * (1 - f);
			g *= 1 - s;
			break;
		case 5:
			g *= 1 - s;
			b *= 1 - s * f;
			break;
		}
	}
	return std::vector<float>{r, g, b};
}
std::vector<float> rgb2hsv(const std::vector<float> &rgb) {
	float r = rgb[0];
	float g = rgb[1];
	float b = rgb[2];
	float max = r > g ? r : g;
	max = max > b ? max : b;
	float min = r < g ? r : g;
	min = min < b ? min : b;
	float h = max - min;
	if (h > 0.0f) {
		if (max == r) {
			h = (g - b) / h;
			if (h < 0.0f) {
				h += 6.0f;
			}
		}
		else if (max == g) {
			h = 2.0f + (b - r) / h;
		}
		else {
			h = 4.0f + (r - g) / h;
		}
	}
	h /= 6.0f;
	float s = (max - min);
	if (max != 0.0f)
		s /= max;
	float v = max;
	return std::vector<float>{h, s, v};
}

void generatePoints_cpu_kernel(double *pts, int npts, int kid, const IFSMap *maps, int nmaps, uint64_t pointgen_seed, int nfrac) {
	// std::random_device seed_gen;
	// std::mt19937 random(seed_gen());
	// std::minstd_rand random(seed_gen());
	// random.seed(pointgen_seed);

	// std::cout<<"nmaps: "<<nmaps<<std::endl;
	std::cout << std::fixed << std::setprecision(17);
	std::mt19937 e2(pointgen_seed);

	// float limit = std::numeric_limits<float>().max()/2.0;
	float limit = 1.69999e+4;  
	// float limit = 1.69999e+38;
	// int print_once_x = 0;
	// int print_once_y = 0;	


	pts[0 + (kid*npts + 0) * 2] = 0.0;
	pts[1 + (kid*npts + 0) * 2] = 0.0;

	double px = 0.0;
	double py = 0.0;

	// #pragma omp parallel for
	for (int pi = 1; pi < npts; ++pi) {  // Start from 1 beacuse 0 needs to be 0.0
		//select map
		const IFSMap *map = &maps[0];
		/////////////////////////// This computes numpy random
		int a = e2() >> 5;
		int b = e2() >> 6;
		const double prob = (a * 67108864.0 + b) / 9007199254740992.0;
		//////////////////////////////////////////////////////////////
		// std::cout << std::fixed << std::setprecision(16) << prob << std::endl;
		// printf("%lf\n",prob);
		double cump = 0;
		for (int i = 0; i < nmaps; ++i) {
			cump += maps[i].p;
			// std::cout<<"cump: "<<cump<<std::endl;
			if (prob < cump) {
				map = &maps[i]; break;
			}
		}
		// std::cout <<"prob: "<< prob << std::endl;		
		// std::cout <<"a:"<< std::fixed << std::setprecision(16) << map->a << std::endl;	
		// std::cout <<"e:"<< std::fixed << std::setprecision(16) << map->e << std::endl;		
		//translate
		double &newnx = pts[0 + (kid*npts + pi) * 2];
		double &newny = pts[1 + (kid*npts + pi) * 2];
		
		float nx = map->a * px + map->b * py + map->e;
		float ny = map->c * px + map->d * py + map->f;
		// nx = map->a * px + map->b * py + map->e;
		// ny = map->c * px + map->d * py + map->f;

		if ((nx > -limit) && (nx < limit) && std::isnormal(nx)){
			// if (nfrac == 203 )std::cout<<"x: "<<nx<<std::endl;
			px = nx;
		}
		else{
			// std::cout<<"limit positive: "<<limit <<std::endl;
			// std::cout<<"limit negative: "<<-limit <<std::endl;
			
			// if (std::isnormal(nx)){
			// 	std::cout<<"Number on X is NORMAL: "<<nx<<std::endl;
			// }
			// else{
			// 	std::cout<<"Number on X is NOT =---------------= NORMAL: "<<nx<<std::endl;
			// }

			// if (print_once_x == 0){
			// std::cout<<"original vector: "<<px<<std::endl;
			// std::cout<<"Beyond double: "<<nx<<std::endl;
			// std::cout<<"index X: "<<pi<<std::endl;
			// break;
			// print_once_x = 1;
			// }
			nx = 0.0;
			px = nx;
		}

		if ((ny > -limit) && (ny < limit) && std::isnormal(ny)){
			// if (nfrac == 203 )std::cout<<"y: "<<ny<<std::endl;
			py = ny;
			
		}
		else{
			// std::cout<<"limit positive: "<<limit<<std::endl;
			// std::cout<<"limit negative: "<<-limit<<std::endl;

			// if (std::isnormal(ny)){
			// 	std::cout<<"Number on Y is NORMAL: "<<ny<<std::endl;
			// }
			// else{
			// 	std::cout<<"Number on Y is NOT =---------------= NORMAL: "<<ny<<std::endl;
			// }


			// if (print_once_y == 0){
			// std::cout<<"original vector: "<<py<<std::endl;
			// std::cout<<"Beyond double: "<<ny<<std::endl;
			// std::cout<<"index Y: "<<pi<<std::endl;
			// break;
			// print_once_y = 1;
			// }
			ny = 0.0;
			py = ny;
		}

		newnx = nx;
		newny = ny;
		// if (std::isnormal(nx) && (nx > (-1)*std::numeric_limits<double>().max()/1000000000000000000 && nx < std::numeric_limits<double>().max()/1000000000000000000)){
		// 	px = nx;
		// }
		// else
		// 	nx = 0.0;

		// if (std::isnormal(ny) && (ny > (-1)*std::numeric_limits<double>().max()/2 && ny < std::numeric_limits<double>().max()/2)){
		// 	py = ny;
		// }
		// else
		// 	nx = 0.0;		
		// px = nx;
		// py = ny;
		// if (nfrac == 2 && pi < 5 )  {
		// 	std::cout<<"x_fromC: "<<pts[0 + (kid*npts + pi-1) * 2]<<std::endl;
		// 	std::cout<<"y_fromC: "<<pts[1 + (kid*npts + pi-1) * 2]<<std::endl;
		// }
		// if (pi == 3){
		// 	exit(0);
		// }

	}
	// exit(0);
}

void generatePoints_cpu(double *pts, int npts, const std::vector<std::vector<IFSMap>> &mapss, uint64_t pointgen_seed, int nfrac) {
	const int ninstances = mapss.size();
	// std::cout<<"ninstances: "<<ninstances<<std::endl;
#ifdef _OPENMP 
#pragma omp parallel for
#endif
	for (int ii = 0; ii < ninstances; ++ii) {
		// std::cout<<"ii: "<<ii<<std::endl;
		generatePoints_cpu_kernel(pts, npts, ii, mapss[ii].data(), mapss[ii].size(), pointgen_seed, nfrac);
	}
}

void renderPoints_cpu_kernel(const double *pts, int npts, int kid, int ninstances, unsigned char *imgs, int width, int height, int patch_mode, int flip_flg, uint64_t patchgen_seed) {
	std::random_device seed_gen;
	// //slow!
	// std::mt19937 randomer(seed_gen());
	// randomer.seed(patchgen_seed);
	// std::uniform_int_distribution<unsigned int> genpp_r(1, (1<<9)-1); //[a,b]
	// auto genpp=[&](){return genpp_r(randomer);}; //[1,511]
	// std::uniform_int_distribution<unsigned int> random_r(0, std::numeric_limits<unsigned int>::max()); //[a,b]
	// auto random=[&](){return random_r(randomer)/(double)random_r.max();}; //[0,1]
	
	//fast!
	// std::minstd_rand randomer(seed_gen());
	std::mt19937 randomer(seed_gen());
	randomer.seed(patchgen_seed);

	auto genpp=[&](){return (unsigned int)(randomer()*(double)(512-1)/(double)((uint64_t)randomer.max()+1)+1);}; //[1, 512]
	auto random=[&](){return randomer()/(double)randomer.max();}; //[0.0, 1.0]

	auto pimg = [&](unsigned char *imgs_, int ii, int r, int c) {return &imgs_[((ii * height + r) * width + c) * 3]; };
	
	unsigned int pap_g = genpp();
	
	//3x3 random dots plain gray
	auto drawPatch_P3RDPG = [&](unsigned char *img, int r, int c) {
		//generate patch pattern
		unsigned int pap = genpp();
		unsigned int pah = 1;
		//uint8*3*9 rgbrgbrgb ※はみ出し…？
		//_m512i mimg=
		//draw patch
		for (int kr = 0; kr < 3; ++kr) {
			for (int kc = 0; kc < 3; ++kc) {
				int nr = r + kr - 3 / 2;
				int nc = c + kc - 3 / 2;
				//check boundary
				if (nr >= 0 && nr < height && nc >= 0 && nc < width) {
					unsigned char *pp = pimg(img, 0, nr, nc);
					const unsigned char cl = (pap&pah) ? 127 : 0;
					pp[0] = cl; pp[1] = cl; pp[2] = cl; //pp[3] = 255;
				}
				pah <<= 1;
			}
		}
	};
	//3x3 all dots plain gray
	auto drawPatch_P3ADPG = [&](unsigned char *img, int r, int c) {
		// printf("From all dots...");
		//draw patch
		for (int kr = 0; kr < 3; ++kr) {
			for (int kc = 0; kc < 3; ++kc) {
				int nr = r + kr - 3 / 2;
				int nc = c + kc - 3 / 2;
				//check boundary
				if (nr >= 0 && nr < height && nc >= 0 && nc < width) {
					unsigned char *pp = pimg(img, 0, nr, nc);
					const unsigned char cl = 127;
					pp[0] = cl; pp[1] = cl; pp[2] = cl; //pp[3] = 255;
				}
			}
		}
	};
	//3x3 random dots phased color
	auto drawPatch_P3RDPC = [&](unsigned char *img, int r, int c, int pi) {
		//generate patch pattern
		unsigned int pap = genpp();
		unsigned int pah = 1;
		//draw patch
		for (int kr = 0; kr < 3; ++kr) {
			for (int kc = 0; kc < 3; ++kc) {
				int nr = r + kr - 3 / 2;
				int nc = c + kc - 3 / 2;
				//check boundary
				if (nr >= 0 && nr < height && nc >= 0 && nc < width) {
					unsigned char *pp = pimg(img, 0, nr, nc);
					auto hsv = hsv2rgb(std::vector<float>{(pi%128)/128.0f, 0.75f, (pah&pap) ? 0.75f : 0.0f});
					pp[0] = hsv[0] * 255; pp[1] = hsv[1] * 255; pp[2] = hsv[2] * 255; //pp[3] = 255;
				}
				pah <<= 1;
			}
		}
	};
	//3x3 all dots random color
	auto drawPatch_P3ADRC = [&](unsigned char *img, int r, int c) {
		//draw patch
		for (int kr = 0; kr < 3; ++kr) {
			for (int kc = 0; kc < 3; ++kc) {
				int nr = r + kr - 3 / 2;
				int nc = c + kc - 3 / 2;
				//check boundary
				if (nr >= 0 && nr < height && nc >= 0 && nc < width) {
					unsigned char *pp = pimg(img, 0, nr, nc);
					auto rgb = hsv2rgb(std::vector<float>{(float)(random()*1.0), (float)(random()*1.0), (float)(random()*0.75)});
					pp[0] = (unsigned char)(rgb[0]*255); pp[1] = (unsigned char)(rgb[1]*255); pp[2] = (unsigned char)(rgb[2]*255); //pp[3] = 255;
				}
			}
		}
	};
	//3x3 all dots random gray
	auto drawPatch_P3ADRG = [&](unsigned char *img, int r, int c) {
		//draw patch
		for (int kr = 0; kr < 3; ++kr) {
			for (int kc = 0; kc < 3; ++kc) {
				int nr = r + kr - 3 / 2;
				int nc = c + kc - 3 / 2;
				//check boundary
				if (nr >= 0 && nr < height && nc >= 0 && nc < width) {
					unsigned char *pp = pimg(img, 0, nr, nc);
					unsigned char cl = (unsigned char)(random()*255);
					pp[0] = pp[1] = pp[2] = cl; //pp[3] = 255;
				}
			}
		}
	};

	//3x3 all random dots random gray
	auto drawPatch_P3RDRG = [&](unsigned char *img, int r, int c) {
		//generate patch pattern
		unsigned int pap = genpp();
		unsigned int pah = 1;
		//draw patch
		for (int kr = 0; kr < 3; ++kr) {
			for (int kc = 0; kc < 3; ++kc) {
				int nr = r + kr - 3 / 2;
				int nc = c + kc - 3 / 2;
				//check boundary
				if (nr >= 0 && nr < height && nc >= 0 && nc < width) {
					unsigned char *pp = pimg(img, 0, nr, nc);
					// const unsigned char cl = (pap&pah) ? 127 : 0;
					unsigned char cl = (pap&pah) ? (unsigned char)(random()*255) : 0;
					pp[0] = pp[1] = pp[2] = cl; //pp[3] = 255;
				}
				pah <<= 1;
			}
		}
	};
	//1x1 all dots plain gray
	auto drawPatch_P1ADPG = [&](unsigned char *img, int r, int c) {
		// printf("From all dots...");
		//draw patch
		for (int kr = 0; kr < 3; ++kr) {
			for (int kc = 0; kc < 3; ++kc) {
				int nr = r + kr - 3 / 2;
				int nc = c + kc - 3 / 2;
				//check boundary
				if (nr >= 0 && nr < height && nc >= 0 && nc < width && kr == 1 && kc == 1) {
					unsigned char *pp = pimg(img, 0, nr, nc);
					const unsigned char cl = 127;
					pp[0] = cl; pp[1] = cl; pp[2] = cl; //pp[3] = 255;
				}
			}
		}
	};
	//3x3 random dots plain gray
	auto drawPatch_P3FDPG = [&](unsigned char *img, int r, int c) {
		//generate patch pattern
		// unsigned int pap = genpp();
		unsigned int pah = 1;
		//uint8*3*9 rgbrgbrgb ※はみ出し…？
		//_m512i mimg=
		//draw patch
		for (int kr = 0; kr < 3; ++kr) {
			for (int kc = 0; kc < 3; ++kc) {
				int nr = r + kr - 3 / 2;
				int nc = c + kc - 3 / 2;
				//check boundary
				if (nr >= 0 && nr < height && nc >= 0 && nc < width) {
					unsigned char *pp = pimg(img, 0, nr, nc);
					const unsigned char cl = (pap_g&pah) ? 127 : 0;
					pp[0] = cl; pp[1] = cl; pp[2] = cl; //pp[3] = 255;
				}
				pah <<= 1;
			}
		}
	};


	//obtain aabb
	double pxmin, pxmax, pymin, pymax;
	for (int pi = 0; pi < npts; ++pi) {
		double px = pts[0 + (kid * npts + pi) * 2];
		double py = pts[1 + (kid * npts + pi) * 2];
		if (pi == 0) { pxmin = pxmax = 0.0; pymin = pymax = 0.0; }
		else {
			//inf対策
			if((px==0 || std::isnormal(px) && (px >= (-1)*std::numeric_limits<float>().max()/2) && (px < std::numeric_limits<float>().max()/2))){
				if (px < pxmin) pxmin = px;
				if (px > pxmax) pxmax = px;
			}
			if((py==0 || std::isnormal(py) && (py >= (-1)*std::numeric_limits<float>().max()/2) && (py < std::numeric_limits<float>().max()/2))){

				if (py < pymin) pymin = py;
				if (py > pymax) pymax = py;
			}


		}
	}

	// if (std::isnormal(pxmax))std::cout<<"limit:"<<std::numeric_limits<double>().max()/100000000000000 <<std::endl;

	// std::cout<<"pxmax:"<<pxmax<<std::endl;
	// std::cout<<"pxmin:"<<pxmin<<std::endl;
	// std::cout<<"pymax:"<<pymax<<std::endl;
	// std::cout<<"pymin:"<<pymin<<std::endl<<std::endl;

	// exit(0);

	double xsize = std::abs(pxmax-pxmin), ysize = std::abs(pymax - pymin);
	// double size = std::max(xsize, ysize);
	double xcenter = (pxmax + pxmin) / 2.0, ycenter = (pymax + pymin) / 2.0;
	// double xcenter = (pxmax + pxmin), ycenter = (pymax + pymin);
	static const int pad_x = 6, pad_y = 6;

	//render points
	for (int pi = 0; pi < npts; ++pi) {
		double px = pts[0 + (kid * npts + pi) * 2];
		double py = pts[1 + (kid * npts + pi) * 2];

		float tx = px,ty = py;
		if (pxmin < 0.0) tx -= pxmin;
		if (pymin < 0.0) ty -= pymin;

		//normalize and quantize with keeping aspect ratio
		int pc = (uint16_t)(((tx) / (float)(pxmax - pxmin)) * (float)(width-2*pad_x)+(float)pad_x);
		int pr = (uint16_t)(((ty) / (float)(pymax - pymin)) * (float)(height-2*pad_y)+(float)pad_y);

		///////////////////////////////////////  Working
		// int pc = (uint16_t)(((px - xcenter) / xsize + 0.5) * (width -2*pad_x) + pad_x);
		// int pr = (uint16_t)(((py - ycenter) / ysize + 0.5) * (height-2*pad_y) + pad_y);

		// if (pi < 5){
		// 	std::cout<<"x_proj:"<<pc<<std::endl;
		// 	std::cout<<"y_proj:"<<pr<<std::endl;
		// 	// std::cout<<"x:"<<tx<<std::endl;
		// 	// std::cout<<"xs:"<<pc<<std::endl;
		// }

		// if (pi == 5) exit(0);
		////////////////////////////////////////
		
		// //normalize and quantize and flip with keeping aspect ration
		// int pc = (int)round(((px - xcenter) / size + 0.5)* (width-2*pad_x)+pad_x);
		// if(flip_flg&1) pc=width-pc-1;
		// int pr = (int)round(((py - ycenter) / size + 0.5)* (height-2*pad_y)+pad_y);
		// if(flip_flg&2) pr=height-pr-1;

		// //normalize and quantize with breaking aspect ratio
		// int pc, pr;
		// int pmx,pmy;
		// //pmx=(px<0)?pxmin:0; //tracing original process //想定外？
		// //pmy=(py<0)?pymin:0; //tracing original process
		// pmx=pxmin;
		// pmy=pymin;
		// pc = (int)round(((double)(px - pmx) / (pxmax-pxmin))* (width-2*pad_x)+pad_x); //アス比も維持してなかった！
		// pr = (int)round(((double)(py - pmy) / (pymax-pymin))* (height-2*pad_y)+pad_y);

		//check boundary
		if (pc >= 0 && pc < width && pr >= 0 && pr < height) {
			switch (patch_mode) {
			case 0:
				drawPatch_P3RDPG(pimg(imgs, kid, 0, 0), pr, pc);
				break;
			case 1:
				drawPatch_P3ADPG(pimg(imgs, kid, 0, 0), pr, pc);
				break;
			case 2:
				drawPatch_P3RDPC(pimg(imgs, kid, 0, 0), pr, pc, pi);
				break;
			case 3:
				drawPatch_P3ADRG(pimg(imgs, kid, 0, 0), pr, pc);
				break;
			case 4:
				drawPatch_P3RDRG(pimg(imgs, kid, 0, 0), pr, pc);
				break;
			case 5:
				drawPatch_P1ADPG(pimg(imgs, kid, 0, 0), pr, pc);
				break;
			case 6:
				drawPatch_P3FDPG(pimg(imgs, kid, 0, 0), pr, pc);
				break;
			}
		}
	}

	//flip on image
	if(flip_flg&1){
		for(int r=0;r<height;++r){
			for(int c=0;c<width/2;++c){
				unsigned char *s = pimg(imgs,kid, r, c);
				int tr = r;
				int tc = width-c-1;
				unsigned char *t = pimg(imgs,kid, tr, tc);
				std::swap(s[0],t[0]);
				std::swap(s[1],t[1]);
				std::swap(s[2],t[2]);
			}
		}
	}
	if(flip_flg&2){
		for(int r=0;r<height/2;++r){
			for(int c=0;c<width;++c){
				unsigned char *s = pimg(imgs,kid, r, c);
				int tr = height-r-1;
				int tc = c;
				unsigned char *t = pimg(imgs,kid, tr, tc);
				std::swap(s[0],t[0]);
				std::swap(s[1],t[1]);
				std::swap(s[2],t[2]);
			}
		}
	}
	

}

void renderPoints_cpu(const double *pts, int npts, int ninstances, unsigned char *imgs, int width, int height, int patch_mode, int flip_flg, uint64_t patchgen_seed) {
#ifdef _OPENMP 
#pragma omp parallel for
#endif
	for (int ii = 0; ii < ninstances; ++ii) {
		renderPoints_cpu_kernel(pts, npts, ii, ninstances, imgs, width, height, patch_mode, flip_flg, patchgen_seed);
	}
}

