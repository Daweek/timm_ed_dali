#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<iostream>
#include<vector>
#include<random>
#include<sstream>
#include<fstream>
#include<iomanip>
#include<regex>
#include<experimental/filesystem>
namespace fs = std::experimental::filesystem;

#include<argagg/argagg.hpp>
#include<opencv2/imgcodecs.hpp>

#include"Fractal2DRenderer_cpu.h"

int countNonZeros(uint8_t *imgs, int width, int height){
    int count=0;
    for(int r=0;r<height;++r){
        for(int c=0;c<width;++c){
            count += (imgs[3*(c+width*r)]>0)?1:0;
        }
    }
    return count;
}

void saveMaps(const char *fname, std::vector<IFSMap> maps){
    FILE *fp=fopen(fname,"w");
	if (!fp) { perror(fname); abort(); }
    for(int i=0;i<maps.size();++i)
        fprintf(fp,"%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e\n",maps[i].a,maps[i].b,maps[i].c,maps[i].d,maps[i].e,maps[i].f,maps[i].p);
    fclose(fp);
}

int main(int argc, const char** argv){
    argagg::parser argparser{{
        //mandatory
        {"fn_prefix",{"--fn_prefix"},"output file prefix including path (mandatory)",1},
        {"iters",{"--iters"},"number of parameter search iteration (either mandatory)", 1},
        {"categories",{"--categories"},"number of valid category count to stop iteration (either mandatory)", 1},

        {"help",{"-h","--help"},"show this message", 0},
        {"paramgen_seed",{"--paramgen_seed"},"random seed for parameter generation", 1},
        {"use_checkpoint",{"--use_checkpoint"},"checkpoint file to use", 1}, //exclusive (priority by order)
        {"pointgen_seed",{"--pointgen_seed"},"random seed for point generation", 1}, //exclusive
        {"width",{"--width"},"internal image width", 1},
        {"height",{"--height"},"internal image height", 1},
        {"npts",{"--npts"},"internal number of points to generate", 1},
        {"thresh",{"--thresh"},"threshold for validation of occupancy on a rendered image", 1},
        {"checkpoint_iters",{"--checkpoint_iters"},"output duration of checkpoint for paramgen random state (0 to disable)", 1},
        {"range_nmaps_min",{"--range_nmaps_min"},"lower range of number of maps", 1},
        {"range_nmaps_max",{"--range_nmaps_max"},"upper range of number of maps", 1},
        {"range_param_min",{"--range_param_min"},"lower range of parameter", 1},
        {"range_param_max",{"--range_param_max"},"upper range of parameter", 1},
        {"p_by_det",{"--p_by_det"},"using determinant of parameter for probability",0},
        {"enable_image_output",{"--enable_image_output"},"flag for category image output", 0},
        
        {"debug",{"--debug"},"debug flag", 0},
    }};
    argagg::parser_results args = argparser.parse(argc, argv);

    if( argc<=1 || args["help"] || !args["fn_prefix"] || (!args["iters"] && !args["categories"]) ){
        std::cout << "usage: " << argv[0] << " --fn_prefix=<path-including-'/'> [--iters=<number> or --categories=<number>] (args...)" << std::endl;
        std::cout << "exmaple: ./Fractal2DCategorySearcher --fn_prefix=./c1kr03/cat_ --cetegories=1000 --thresh=0.3" << std::endl;
        std::cout << "Required previously to create target output directory." << std::endl;
        std::cout << "options:" << std::endl;
        std::cout << argparser;
        exit(0);
    }
    //else{std::cout<<"nope"<<std::endl;}

    std::string fn_prefix = args["fn_prefix"].as<std::string>();

    uint32_t paramgen_seed = args["paramgen_seed"].as<uint32_t>(100);
    uint32_t pointgen_seed = args["pointgen_seed"].as<uint32_t>(100);
    int width = args["width"].as<int>(256);
    int height = args["height"].as<int>(256);
    int npts = args["npts"].as<int>(100000);
    unsigned int iters,categories;
    int iters_keta = 0;
    if(args["iters"]){
        iters = (unsigned int)args["iters"].as<int>();
        categories = (unsigned int)args["categories"].as<int>(-1);
        iters_keta = (int)floor(log(iters)/log(10))+1;
    } else {
        iters = (unsigned int)((int)-1);
        categories = (unsigned int)args["categories"].as<int>();
        iters_keta = (int)floor(log(categories)/log(10))+1;
    }
    double validThresh = args["thresh"].as<double>(0.2);
    int validThreshCnt = (int)floor(validThresh*width*height);
    int checkpoint_iters = args["checkpoint_iters"].as<int>(0);
    bool isDebug = args["debug"];
    bool enable_image_output = args["enable_image_output"];

    double
        prange_min_a=-1,
        prange_max_a=1,
        prange_min_b=-1,
        prange_max_b=1,
        prange_min_c=-1,
        prange_max_c=1,
        prange_min_d=-1,
        prange_max_d=1,
        prange_min_e=-1,
        prange_max_e=1,
        prange_min_f=-1,
        prange_max_f=1,
        prange_min_p=0,
        prange_max_p=1;
    prange_min_a = prange_min_b = prange_min_c = prange_min_d = prange_min_e = prange_min_f
        = args["range_param_min"].as<double>(-1.0);
    prange_max_a = prange_max_b = prange_max_c = prange_max_d = prange_max_e = prange_max_f
        = args["range_param_max"].as<double>(1.0);

    int range_nmaps_min=args["range_nmaps_min"].as<int>(2), range_nmaps_max=args["range_nmaps_min"].as<int>(8);

    int start_iter=0;

    std::mt19937 random;
    if(args["use_checkpoint"]){
        //deserialize random state from file
        std::string checkpoint_fn=args["use_checkpoint"].as<std::string>();
        std::ifstream fin(checkpoint_fn,std::ios::binary);
        fin >> random;
        //parsing start iter from filename
        std::regex re("^.*_0*(\\d+)_\\d+_checkpoint_rng.bin$");
        start_iter = std::stoi(std::regex_replace(checkpoint_fn, re, "$1")) + 1;
    }
    else
        random.seed(paramgen_seed);

    int validCount=0;

    for(int iter=start_iter ; iter < iters ; ++iter){
        // randomly select number of map
        int nmaps = (int)floor((uint64_t)random()* (range_nmaps_max-range_nmaps_min) / (double)((uint64_t)random.max()+1)) + range_nmaps_min;

        // randomly generate parameters of maps
        std::vector<std::vector<IFSMap>> mapss(1, std::vector<IFSMap>(nmaps));
        double sump=0;
        for(int im=0;im<nmaps;++im){
            float a,b,c,d,e,f,p;
            a = random() * (prange_max_a-prange_min_a) / (double)random.max() + prange_min_a;
            b = random() * (prange_max_b-prange_min_b) / (double)random.max() + prange_min_b;
            c = random() * (prange_max_c-prange_min_c) / (double)random.max() + prange_min_c;
            d = random() * (prange_max_d-prange_min_d) / (double)random.max() + prange_min_d;
            e = random() * (prange_max_e-prange_min_e) / (double)random.max() + prange_min_e;
            f = random() * (prange_max_f-prange_min_f) / (double)random.max() + prange_min_f;
            p = random() * (prange_max_p-prange_min_p) / (double)random.max() + prange_min_p;
            mapss[0][im] = IFSMap{a,b,c,d,e,f,p};
            sump+=p;
        }
        //normalization for probability
        for(int im=0;im<nmaps;++im) mapss[0][im].p/=sump;
        if(args["p_by_det"]){
            std::vector<double> dets(nmaps);
            double dsum=0;
            for(int i=0 ; i<nmaps ; ++i)
                dsum += dets[i] = fabs(mapss[0][i].a*mapss[0][i].d-mapss[0][i].b*mapss[0][i].c);
            for(int i=0 ; i<nmaps ; ++i)
                mapss[0][i].p = dets[i] / dsum;
        }

        // generate points and render image
        int ninstances = mapss.size();
        std::vector<double> pts(ninstances * npts * 2, 0);
        std::vector<uint8_t> imgs(ninstances * width * height * 3, 0);

        generatePoints_cpu(pts.data(), npts, mapss, pointgen_seed);
        renderPoints_patched_cpu(pts.data(), npts, ninstances, imgs.data(), width, height, 0, 1, 1);

        //check validity then write to file
        int pcnt = countNonZeros(imgs.data(), width, height);
        if(pcnt > validThreshCnt){
            //valid
            //save as csv
            {
                std::ostringstream sout;
                sout << fn_prefix;
                sout << std::setfill('0') << std::right << std::setw(iters_keta) << validCount;
                sout << std::resetiosflags(std::ios_base::floatfield);
                saveMaps((sout.str()+".csv").c_str(), mapss[0]);

                if(enable_image_output){
                    cv::Mat cimg(height, width, CV_8UC3, imgs.data());
                    cv::imwrite(sout.str()+"_img.png", cimg);
                }
            }
            ++validCount;
        }

        //save submaterial for debug
        if(isDebug){
            std::ostringstream sout;
            sout << fn_prefix;
            sout << std::setfill('0') << std::right << std::setw(iters_keta) << iter;
            //sout << std::resetiosflags(std::ios_base::floatfield);
            {
                FILE *ofp=fopen((sout.str()+"_pts.bin").c_str(),"wb");
                fwrite(pts.data(), sizeof(double), 1*npts*2, ofp);
                fclose(ofp);
            }
            {
                FILE *ofp=fopen((sout.str()+"_imgs.bin").c_str(),"wb");
                fwrite(imgs.data(), sizeof(uint8_t), 1*width*height*3, ofp);
                fclose(ofp);
            }
        }

        //saving checkpoint
        if(checkpoint_iters>0 && ((iter+1)%checkpoint_iters==0)){
            std::ostringstream sout;
            sout << fn_prefix;
            sout << std::setfill('0') << std::right << std::setw(iters_keta) << iter;
            sout << std::resetiosflags(std::ios_base::floatfield);
            sout << "_";
            sout << std::setfill('0') << std::right << std::setw(iters_keta) << validCount;
            std::ofstream fout(sout.str()+"_checkpoint_rng.bin", std::ios::binary);
            fout << random;
        }

        //break on reaching limit
        if(validCount>=categories)
            break;
    }

    return 0;
}