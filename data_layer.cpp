/**************
dataLayer作为整个网络的输入层，

数据从leveldb中取。leveldb的数据是通过图片转换过来的。

网络建立的时候，

datalayer主要是负责设置一些参数，比如batchsize，channels，height，width等。

这次会通过读leveldb一个数据块来获取这些信息。

然后启动一个线程来预先从leveldb拉取一批数据，这些数据是图像数据和图像标签。



正向传播的时候，

datalayer就把预先拉取好数据拷贝到指定的cpu或者gpu的内存。

然后启动新线程再预先拉取数据，这些数据留到下一次正向传播使用。
****************/


// Copyright 2013 Yangqing Jia  
  
#include <stdint.h>  
#include <leveldb/db.h>  
#include <pthread.h>  
  
#include <string>  
#include <vector>  
  
#include "caffe/layer.hpp"  
#include "caffe/util/io.hpp"  
#include "caffe/vision_layers.hpp"  
  
using std::string;  
  
namespace caffe {  
  
template <typename Dtype>  
void* DataLayerPrefetch(void* layer_pointer) {  
  CHECK(layer_pointer);  
  DataLayer<Dtype>* layer = reinterpret_cast<DataLayer<Dtype>*>(layer_pointer);  
  CHECK(layer);  
  Datum datum;  
  CHECK(layer->prefetch_data_);  
  Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();//数据  
  Dtype* top_label = layer->prefetch_label_->mutable_cpu_data();//标签  
  const Dtype scale = layer->layer_param_.scale();  
  const int batchsize = layer->layer_param_.batchsize();  
  const int cropsize = layer->layer_param_.cropsize();  
  const bool mirror = layer->layer_param_.mirror();  
  
  if (mirror && cropsize == 0) {//当前实现需要同时设置mirror和cropsize  
    LOG(FATAL) << "Current implementation requires mirror and cropsize to be "  
        << "set at the same time.";  
  }  
  // datum scales  
  const int channels = layer->datum_channels_;  
  const int height = layer->datum_height_;  
  const int width = layer->datum_width_;  
  const int size = layer->datum_size_;  
  const Dtype* mean = layer->data_mean_.cpu_data();  
  for (int itemid = 0; itemid < batchsize; ++itemid) {//每一批数据的数量是batchsize，一个循环拉取一张？  
    // get a blob  
    CHECK(layer->iter_);  
    CHECK(layer->iter_->Valid());  
    datum.ParseFromString(layer->iter_->value().ToString());//利用迭代器拉取下一批数据  
    const string& data = datum.data();  
    if (cropsize) {//如果需要裁剪  
      CHECK(data.size()) << "Image cropping only support uint8 data";  
      int h_off, w_off;  
      // We only do random crop when we do training.  
      //只是在训练阶段做随机裁剪  
      if (Caffe::phase() == Caffe::TRAIN) {  
        // NOLINT_NEXT_LINE(runtime/threadsafe_fn)  
        h_off = rand() % (height - cropsize);  
        // NOLINT_NEXT_LINE(runtime/threadsafe_fn)  
        w_off = rand() % (width - cropsize);  
      } else {//测试阶段固定裁剪  
        h_off = (height - cropsize) / 2;  
        w_off = (width - cropsize) / 2;  
      }  
      // NOLINT_NEXT_LINE(runtime/threadsafe_fn)  
      //怎么感觉下面两种情况的代码是一样的？  
      if (mirror && rand() % 2) {  
        // Copy mirrored version  
        for (int c = 0; c < channels; ++c) {  
          for (int h = 0; h < cropsize; ++h) {  
            for (int w = 0; w < cropsize; ++w) {  
              top_data[((itemid * channels + c) * cropsize + h) * cropsize  
                       + cropsize - 1 - w] =  
                  (static_cast<Dtype>(  
                      (uint8_t)data[(c * height + h + h_off) * width  
                                    + w + w_off])  
                    - mean[(c * height + h + h_off) * width + w + w_off])  
                  * scale;  
            }  
          }  
        }  
      } else {  
        // Normal copy  
        for (int c = 0; c < channels; ++c) {  
          for (int h = 0; h < cropsize; ++h) {  
            for (int w = 0; w < cropsize; ++w) {  
              top_data[((itemid * channels + c) * cropsize + h) * cropsize + w]  
                  = (static_cast<Dtype>(  
                      (uint8_t)data[(c * height + h + h_off) * width  
                                    + w + w_off])  
                     - mean[(c * height + h + h_off) * width + w + w_off])  
                  * scale;  
            }  
          }  
        }  
      }  
    } else {//如果不需要裁剪  
      // we will prefer to use data() first, and then try float_data()  
      //我们优先考虑data()，然后float_data()  
      if (data.size()) {  
        for (int j = 0; j < size; ++j) {  
          top_data[itemid * size + j] =  
              (static_cast<Dtype>((uint8_t)data[j]) - mean[j]) * scale;  
        }  
      } else {  
        for (int j = 0; j < size; ++j) {  
          top_data[itemid * size + j] =  
              (datum.float_data(j) - mean[j]) * scale;  
        }  
      }  
    }  
  
    top_label[itemid] = datum.label();  
    // go to the next iter  
    layer->iter_->Next();  
    if (!layer->iter_->Valid()) {  
      // We have reached the end. Restart from the first.  
      DLOG(INFO) << "Restarting data prefetching from start.";  
      layer->iter_->SeekToFirst();  
    }  
  }  
  
  return reinterpret_cast<void*>(NULL);  
}  
  
template <typename Dtype>  
DataLayer<Dtype>::~DataLayer<Dtype>() {  
  // Finally, join the thread  
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";  
}  
  
template <typename Dtype>  
void DataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,  
      vector<Blob<Dtype>*>* top) {  
  CHECK_EQ(bottom.size(), 0) << "Data Layer takes no input blobs.";  
  CHECK_EQ(top->size(), 2) << "Data Layer takes two blobs as output.";  
  // Initialize the leveldb  
  leveldb::DB* db_temp;  
  leveldb::Options options;  
  options.create_if_missing = false;  
  options.max_open_files = 100;  
  LOG(INFO) << "Opening leveldb " << this->layer_param_.source();  
  leveldb::Status status = leveldb::DB::Open(  
      options, this->layer_param_.source(), &db_temp);  
  CHECK(status.ok()) << "Failed to open leveldb "  
      << this->layer_param_.source() << std::endl << status.ToString();  
  db_.reset(db_temp);  
  iter_.reset(db_->NewIterator(leveldb::ReadOptions()));//通过迭代器来操纵leveldb  
  iter_->SeekToFirst();  
  // Check if we would need to randomly skip a few data points  
  //是否要随机跳过一些数据  
  if (this->layer_param_.rand_skip()) {  
    // NOLINT_NEXT_LINE(runtime/threadsafe_fn)  
    unsigned int skip = rand() % this->layer_param_.rand_skip();  
    LOG(INFO) << "Skipping first " << skip << " data points.";  
    while (skip-- > 0) {//循环次数  
      iter_->Next();  
      if (!iter_->Valid()) {  
        iter_->SeekToFirst();  
      }  
    }  
  }  
  // Read a data point, and use it to initialize the top blob.  
  //读取一个数据点，用来初始化topblob。所谓初始化，只要是指reshape。  
  //可以观察到下面iter_调用调用next。所以这次读取只是用来读取出来channels等参数的，不作处理。  
  Datum datum;  
  datum.ParseFromString(iter_->value().ToString());//利用迭代器读取第一个数据点  
  // image图像数据  
  int cropsize = this->layer_param_.cropsize();//裁剪大小  
  if (cropsize > 0) {//需要裁剪  
    (*top)[0]->Reshape(  
        this->layer_param_.batchsize(), datum.channels(), cropsize, cropsize);  
    prefetch_data_.reset(new Blob<Dtype>(  
        this->layer_param_.batchsize(), datum.channels(), cropsize, cropsize));  
  } else {//不需要裁剪  
    (*top)[0]->Reshape(  
        this->layer_param_.batchsize(), datum.channels(), datum.height(),  
        datum.width());  
    prefetch_data_.reset(new Blob<Dtype>(  
        this->layer_param_.batchsize(), datum.channels(), datum.height(),  
        datum.width()));  
  }  
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","  
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","  
      << (*top)[0]->width();  
  // label标签数据  
  (*top)[1]->Reshape(this->layer_param_.batchsize(), 1, 1, 1);  
  prefetch_label_.reset(  
      new Blob<Dtype>(this->layer_param_.batchsize(), 1, 1, 1));  
  // datum size  
  datum_channels_ = datum.channels();  
  datum_height_ = datum.height();  
  datum_width_ = datum.width();  
  datum_size_ = datum.channels() * datum.height() * datum.width();  
  CHECK_GT(datum_height_, cropsize);  
  CHECK_GT(datum_width_, cropsize);  
  // check if we want to have mean是否要减去均值  
  if (this->layer_param_.has_meanfile()) {  
    BlobProto blob_proto;  
    LOG(INFO) << "Loading mean file from" << this->layer_param_.meanfile();  
    ReadProtoFromBinaryFile(this->layer_param_.meanfile().c_str(), &blob_proto);  
    data_mean_.FromProto(blob_proto);  
    CHECK_EQ(data_mean_.num(), 1);  
    CHECK_EQ(data_mean_.channels(), datum_channels_);  
    CHECK_EQ(data_mean_.height(), datum_height_);  
    CHECK_EQ(data_mean_.width(), datum_width_);  
  } else {  
    // Simply initialize an all-empty mean.  
    data_mean_.Reshape(1, datum_channels_, datum_height_, datum_width_);  
  }  
  // Now, start the prefetch thread. Before calling prefetch, we make two  
  // cpu_data calls so that the prefetch thread does not accidentally make  
  // simultaneous cudaMalloc calls when the main thread is running. In some  
  // GPUs this seems to cause failures if we do not so.  
  prefetch_data_->mutable_cpu_data();  
  prefetch_label_->mutable_cpu_data();  
  data_mean_.cpu_data();  
  DLOG(INFO) << "Initializing prefetch";  
  CHECK(!pthread_create(&thread_, NULL, DataLayerPrefetch<Dtype>,  
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";  
  DLOG(INFO) << "Prefetch initialized.";  
}  
  
template <typename Dtype>  
void DataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,  
      vector<Blob<Dtype>*>* top) {  
  // First, join the thread 等待线程结束  
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";  
  // Copy the data拷贝数据到top，即该层的输出  
  memcpy((*top)[0]->mutable_cpu_data(), prefetch_data_->cpu_data(),  
      sizeof(Dtype) * prefetch_data_->count());  
  memcpy((*top)[1]->mutable_cpu_data(), prefetch_label_->cpu_data(),  
      sizeof(Dtype) * prefetch_label_->count());  
  // Start a new prefetch thread启动新线程拉取下一批数据  
  CHECK(!pthread_create(&thread_, NULL, DataLayerPrefetch<Dtype>,  
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";  
}  
  
// The backward operations are dummy - they do not carry any computation.  
template <typename Dtype>  
Dtype DataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,  
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {  
  return Dtype(0.);  
}  
  
INSTANTIATE_CLASS(DataLayer);  
  
}  // namespace caffe  
