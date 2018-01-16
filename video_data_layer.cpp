/*
http://blog.csdn.net/u014381600/article/details/54287312
*/
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"  //caffe-action/include/caffe下的data_layer.hpp头文件，包含了数据层父类和各个子类，包括VideoDataLayer
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#ifdef USE_MPI
#include "mpi.h"
#include <boost/filesystem.hpp>
using namespace boost::filesystem;
#endif

namespace caffe{
template <typename Dtype>
VideoDataLayer<Dtype>:: ~VideoDataLayer<Dtype>(){
	this->JoinPrefetchThread();
}

template <typename Dtype>
void VideoDataLayer<Dtype>:: DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
        // 这里通过接口函数读取,video_data_param其实就是该层protocol文件中的二级层属性名
	const int new_height  = this->layer_param_.video_data_param().new_height(); //new_height:图像resize之后的height，默认为0，表示忽略
	const int new_width  = this->layer_param_.video_data_param().new_width(); //new_width：图像resize之后的width，默认为0，表示忽略
	const int new_length  = this->layer_param_.video_data_param().new_length();//一个视频最小片段包含几张图片文件，帧数？？？？？
	const int num_segments = this->layer_param_.video_data_param().num_segments(); //segment的数量
	const string& source = this->layer_param_.video_data_param().source();
         
        // 和imagelayer基本一致
	LOG(INFO) << "Opening file: " << source;
	std:: ifstream infile(source.c_str());  //c_str()函数返回一个指向正规c字符串的指针,内容和string类的本身对象是一样的
	string filename;
	int label;
	int length;
	while (infile >> filename >> length >> label){
		lines_.push_back(std::make_pair(filename,label));
		lines_duration_.push_back(length);
	}
        //区别在于对头文件中几个参数的赋值
	if (this->layer_param_.video_data_param().shuffle()){
		const unsigned int prefectch_rng_seed = caffe_rng_rand();
		prefetch_rng_1_.reset(new Caffe::RNG(prefectch_rng_seed));
		prefetch_rng_2_.reset(new Caffe::RNG(prefectch_rng_seed));
		ShuffleVideos();
	}
        
        // 然后就是对输入文件的组织顺序，文件名|视频帧数|类别
        // 日志信息，lines保存的是样本文件名     
	LOG(INFO) << "A total of " << lines_.size() << " videos.";
	lines_id_ = 0;

	//check name patter
        // 名字模式，tsn中定义为img_%05d.jpg
	if (this->layer_param_.video_data_param().name_pattern() == ""){
		if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_RGB){
			name_pattern_ = "image_%04d.jpg";
		}else if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_FLOW){
			name_pattern_ = "flow_%c_%04d.jpg";
		}
	}else{
		name_pattern_ = this->layer_param_.video_data_param().name_pattern();
	}

	Datum datum;  ////定义存储数据
        bool is_color = !this->layer_param_.video_data_param().grayscale();
	const unsigned int frame_prefectch_rng_seed = caffe_rng_rand();
	frame_prefetch_rng_.reset(new Caffe::RNG(frame_prefectch_rng_seed));
	int average_duration = (int) lines_duration_[lines_id_]/num_segments; //dddddddddddddddddddd
	vector<int> offsets;
	for (int i = 0; i < num_segments; ++i){
		caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
		int offset = (*frame_rng)() % (average_duration - new_length + 1);
		offsets.push_back(offset+i*average_duration);
	}
        // 读取文件到datum中,ReadSegmentFlowToDatum等函数定义在io.cpp中
        // 如果是光流模式 ReadSegmentFlowToDatum()函数将图片读入到结构体当中
        // 以函数ReadSegmentRGBToDatum为例，定义在io.cpp，在caffe-action/src/caffe/util中,加载图片到数据库的很多底层函数都在这里定义。
	if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_FLOW)
		CHECK(ReadSegmentFlowToDatum(lines_[lines_id_].first, lines_[lines_id_].second,
									 offsets, new_height, new_width, new_length, &datum, name_pattern_.c_str()));
	else
		CHECK(ReadSegmentRGBToDatum(lines_[lines_id_].first, lines_[lines_id_].second,
									offsets, new_height, new_width, new_length, &datum, is_color, name_pattern_.c_str()));
	const int crop_size = this->layer_param_.transform_param().crop_size();
	const int batch_size = this->layer_param_.video_data_param().batch_size();   //像这里用到的诸多接口，有的就是从proto中编译生成的
	if (crop_size > 0){
		top[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
		this->prefetch_data_.Reshape(batch_size, datum.channels(), crop_size, crop_size);
	} else {
		top[0]->Reshape(batch_size, datum.channels(), datum.height(), datum.width());
		this->prefetch_data_.Reshape(batch_size, datum.channels(), datum.height(), datum.width());
	}
	LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();

	top[1]->Reshape(batch_size, 1, 1, 1);
	this->prefetch_label_.Reshape(batch_size, 1, 1, 1);

	vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
	this->transformed_data_.Reshape(top_shape);
}

template <typename Dtype>
void VideoDataLayer<Dtype>::ShuffleVideos(){
	caffe::rng_t* prefetch_rng1 = static_cast<caffe::rng_t*>(prefetch_rng_1_->generator());
	caffe::rng_t* prefetch_rng2 = static_cast<caffe::rng_t*>(prefetch_rng_2_->generator());
	shuffle(lines_.begin(), lines_.end(), prefetch_rng1);
	shuffle(lines_duration_.begin(), lines_duration_.end(),prefetch_rng2);
}

template <typename Dtype>
void VideoDataLayer<Dtype>::InternalThreadEntry(){

	Datum datum;
	CHECK(this->prefetch_data_.count());
	Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
	Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
	VideoDataParameter video_data_param = this->layer_param_.video_data_param();
	const int batch_size = video_data_param.batch_size();
	const int new_height = video_data_param.new_height();
	const int new_width = video_data_param.new_width();
	const int new_length = video_data_param.new_length();
	const int num_segments = video_data_param.num_segments();
	const int lines_size = lines_.size();

        bool is_color = !this->layer_param_.video_data_param().grayscale();
        // 循环的凑够batch
	for (int item_id = 0; item_id < batch_size; ++item_id){
		CHECK_GT(lines_size, lines_id_);
		vector<int> offsets;
		int average_duration = (int) lines_duration_[lines_id_] / num_segments;
                // 循环num_segments次，每一次采样一个snippet
		for (int i = 0; i < num_segments; ++i){
                        // 如果是训练阶段，则
			if (this->phase_==TRAIN){
				if (average_duration >= new_length){
					caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
					int offset = (*frame_rng)() % (average_duration - new_length + 1);
					offsets.push_back(offset+i*average_duration);
				} else {
					offsets.push_back(0);
				}
			} else{
				if (average_duration >= new_length)
				offsets.push_back(int((average_duration-new_length+1)/2 + i*average_duration));
				else
				offsets.push_back(0);
			}
		}
		if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_FLOW){
			if(!ReadSegmentFlowToDatum(lines_[lines_id_].first, lines_[lines_id_].second,
									   offsets, new_height, new_width, new_length, &datum, name_pattern_.c_str())) {
				continue;
			}
		} else{
			if(!ReadSegmentRGBToDatum(lines_[lines_id_].first, lines_[lines_id_].second,
									  offsets, new_height, new_width, new_length, &datum, is_color, name_pattern_.c_str())) {
				continue;
			}
		}

		int offset1 = this->prefetch_data_.offset(item_id);
    	        this->transformed_data_.set_cpu_data(top_data + offset1);
		this->data_transformer_->Transform(datum, &(this->transformed_data_));
		top_label[item_id] = lines_[lines_id_].second;
		//LOG()

		//next iteration
		lines_id_++;
		if (lines_id_ >= lines_size) {
			DLOG(INFO) << "Restarting data prefetching from start.";
			lines_id_ = 0;
			if(this->layer_param_.video_data_param().shuffle()){
				ShuffleVideos();
			}
		}
	}
}

INSTANTIATE_CLASS(VideoDataLayer);
REGISTER_LAYER_CLASS(VideoData);
}
