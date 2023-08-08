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
#include<functional>
#include<experimental/filesystem>
namespace fs = std::experimental::filesystem;

#include<argagg/argagg.hpp>
#include<opencv2/imgcodecs.hpp>

#include"Fractal2DRenderer_cpu.h"

class STAT{
public:
    int depth;
    std::vector<int> selected;
    STAT(int _depth,const std::vector<int> &_selected):depth(_depth),selected(_selected){}
    
};

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
    const argagg::parser argparser{{
        //mandatory
        {"fn_prefix",{"--fn_prefix"},"output file prefix including path (mandatory)",1},
        {"grid_span",{"--grid_span"},"grid span (mandatory)",1},
        {"grid_perturbate_range", {"--grid_perturbate_range"}, "perturbation range of grid (mandatory)", 1},
        {"grid_perturbate_type", {"--grid_perturbate_type"}, "perturbation random distribution type (uniform, normal) (mandatory)", 1},

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
    const argagg::parser_results args = argparser.parse(argc, argv);

    if( argc<=1 || args["help"] || !args["grid_span"] || !args["grid_perturbate_range"] || !args["grid_perturbate_type"] ){
        std::cout << "usage: " << argv[0] << " --fn_prefix=<path-including-'/'> --grid_span=<number> --grid_perturbate_range=<number> --grid_pertubate_type=[uniform|normal] (args...)" << std::endl;
        std::cout << "exmaple: " << argv[0] << " --fn_prefix=./c1kr03/cat_ --grid_span=0.1 --grid_perturbate_range=0.005 --grid_perturbate_type=uniform" << std::endl;
        std::cout << "Required previously to create target output directory." << std::endl;
        std::cout << "options:" << std::endl;
        std::cout << argparser;
        exit(0);
    }
    //else{std::cout<<"nope"<<std::endl;}

    const std::string fn_prefix = args["fn_prefix"].as<std::string>();

    const uint32_t paramgen_seed = args["paramgen_seed"].as<uint32_t>(100);
    const uint32_t pointgen_seed = args["pointgen_seed"].as<uint32_t>(100);
    const int width = args["width"].as<int>(256);
    const int height = args["height"].as<int>(256);
    const int npts = args["npts"].as<int>(100000);
    const double validThresh = args["thresh"].as<double>(0.2);
    const int validThreshCnt = (int)floor(validThresh*width*height);
    const int checkpoint_iters = args["checkpoint_iters"].as<int>(0);
    const bool enable_image_output = args["enable_image_output"];
    const bool p_by_det = args["p_by_det"];
    const bool debugFlg = args["debug"];

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

    int range_nmaps_min=args["range_nmaps_min"].as<int>(2), range_nmaps_max=args["range_nmaps_max"].as<int>(8);

    double
        grid_span_a = 0.1,
        grid_span_b = 0.1,
        grid_span_c = 0.1,
        grid_span_d = 0.1,
        grid_span_e = 0.1,
        grid_span_f = 0.1;
    grid_span_a = grid_span_b = grid_span_c = grid_span_d = grid_span_e = grid_span_f
        = args["grid_span"].as<double>(0.1);

    int grid_count_a,grid_count_b,grid_count_c,grid_count_d,grid_count_e,grid_count_f;
    grid_count_a = (int)floor((prange_max_a-prange_min_a)/grid_span_a);
    grid_count_b = (int)floor((prange_max_b-prange_min_b)/grid_span_b);
    grid_count_c = (int)floor((prange_max_c-prange_min_c)/grid_span_c);
    grid_count_d = (int)floor((prange_max_d-prange_min_d)/grid_span_d);
    grid_count_e = (int)floor((prange_max_e-prange_min_e)/grid_span_e);
    grid_count_f = (int)floor((prange_max_f-prange_min_f)/grid_span_f);
    int grid_count = grid_count_a*grid_count_b*grid_count_c*grid_count_d*grid_count_e*grid_count_f;

    int iters_keta = (int)floor(log(grid_count*(range_nmaps_max-range_nmaps_min+1))/log(10))+1;

    double
        grid_perturbate_range_a = 0.01,
        grid_perturbate_range_b = 0.01,
        grid_perturbate_range_c = 0.01,
        grid_perturbate_range_d = 0.01,
        grid_perturbate_range_e = 0.01,
        grid_perturbate_range_f = 0.01;
    grid_perturbate_range_a = grid_perturbate_range_b = grid_perturbate_range_c = grid_perturbate_range_d = grid_perturbate_range_e = grid_perturbate_range_f
        = args["grid_perturbate_range"].as<double>(0.01);
    std::string grid_perturbate_type = args["grid_perturbate_type"].as<std::string>("uniform");

    bool doGridPerturbation = !(
        grid_perturbate_range_a == 0 && 
        grid_perturbate_range_b == 0 && 
        grid_perturbate_range_c == 0 && 
        grid_perturbate_range_d == 0 && 
        grid_perturbate_range_e == 0 && 
        grid_perturbate_range_f == 0);

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

    std::uniform_real_distribution<double> gridranddist_uniform(-1, 1);
    std::normal_distribution<double> gridranddist_normal(0, 1);

    int validCount=0;

    for(int nmaps=range_nmaps_min ; nmaps<=range_nmaps_max ; ++nmaps){
        std::function<void(STAT)> dfs_rec = [&](STAT s){
            if(s.depth==nmaps-1){
                //leaf
                std::vector<IFSMap> maps(nmaps);
                double sump=0;
                for(int im=0;im<nmaps;++im){
                    int gi = s.selected[im];
                    int ai = gi%grid_count_a;
                    int bi = gi/grid_count_a%grid_count_b;
                    int ci = gi/grid_count_a/grid_count_b%grid_count_c;
                    int di = gi/grid_count_a/grid_count_b/grid_count_c%grid_count_d;
                    int ei = gi/grid_count_a/grid_count_b/grid_count_c/grid_count_d%grid_count_e;
                    int fi = gi/grid_count_a/grid_count_b/grid_count_c/grid_count_d/grid_count_e%grid_count_f;
                    double a = ai*grid_span_a + prange_min_a;
                    double b = bi*grid_span_b + prange_min_b;
                    double c = ci*grid_span_c + prange_min_c;
                    double d = di*grid_span_d + prange_min_d;
                    double e = ei*grid_span_e + prange_min_e;
                    double f = fi*grid_span_f + prange_min_f;
                    if(doGridPerturbation){
                        if(grid_perturbate_type=="uniform"){
                            a += gridranddist_uniform(random) * grid_perturbate_range_a;
                            b += gridranddist_uniform(random) * grid_perturbate_range_b;
                            c += gridranddist_uniform(random) * grid_perturbate_range_c;
                            d += gridranddist_uniform(random) * grid_perturbate_range_d;
                            e += gridranddist_uniform(random) * grid_perturbate_range_e;
                            f += gridranddist_uniform(random) * grid_perturbate_range_f;
                        }else if(grid_perturbate_type=="normal"){
                            a += gridranddist_normal(random) * grid_perturbate_range_a;
                            b += gridranddist_normal(random) * grid_perturbate_range_b;
                            c += gridranddist_normal(random) * grid_perturbate_range_c;
                            d += gridranddist_normal(random) * grid_perturbate_range_d;
                            e += gridranddist_normal(random) * grid_perturbate_range_e;
                            f += gridranddist_normal(random) * grid_perturbate_range_f;
                        }else{
                            throw(std::invalid_argument("irregal grid perturbation type"));
                        }
                    }
                    double p;
                    if(p_by_det) p = fabs(a*d - b*c);
                    else p = ((double)random()-random.min())/(random.max()-random.min());
                    maps[im] = IFSMap{a, b, c, d, e, f, p};
                    sump += p;

                    if(debugFlg)
                        printf("%g %g %g %g %g %g %g\n",a,b,c,d,e,f,p);
                }
                //printf("\n");
                for(int im=0;im<nmaps;++im) maps[im].p /= sump;

                std::vector<std::vector<IFSMap>> mapss{maps};
                std::vector<double> pts(npts*2, 0);
                std::vector<uint8_t> imgs(width*height*3, 0);
                generatePoints_cpu(pts.data(), npts, mapss, pointgen_seed);
                renderPoints_patched_cpu(pts.data(), npts, 1, imgs.data(), width, height, 0, 1, 1);
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
                    if(debugFlg)
                        printf("valid %d\n", validCount);
                }
                else{
                    if(debugFlg)
                        printf("invalid %d\n", validCount);
                }
            }
            else{
                for(int gi=((s.depth>=0)?s.selected[s.depth]:0);gi<grid_count;++gi){
                    std::vector<int> nselected=s.selected;
                    nselected[s.depth+1]=gi;
                    dfs_rec(STAT(s.depth+1, nselected));
                }
            }
        };
        STAT _s(-1, std::vector<int>(nmaps,-1));
        dfs_rec(_s);
    }

    // //saving checkpoint
    // if(checkpoint_iters>0 && ((iter+1)%checkpoint_iters==0)){
    //     std::ostringstream sout;
    //     sout << fn_prefix;
    //     sout << std::setfill('0') << std::right << std::setw(iters_keta) << iter;
    //     sout << std::resetiosflags(std::ios_base::floatfield);
    //     sout << "_";
    //     sout << std::setfill('0') << std::right << std::setw(iters_keta) << validCount;
    //     std::ofstream fout(sout.str()+"_checkpoint_rng.bin", std::ios::binary);
    //     fout << random;
    // }

    return 0;
}