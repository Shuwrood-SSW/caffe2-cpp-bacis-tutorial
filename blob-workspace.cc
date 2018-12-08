#include <list>
#include <caffe2/core/init.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/operator_gradient.h>
#include <caffe2/proto/caffe2.pb.h>
namespace caffe2 {
    void print( Blob* blob, const std::string& name) {
        Tensor* tensor = BlobGetMutableTensor(blob, caffe2::DeviceType::CPU);
        const auto& data = tensor->data<float>();
        std::cout << name << "(" << tensor->dims() << "): " 
            << std::vector<float>(data, data+tensor->size())
            << std::endl;
    }

    void run() {

   
        // define workspace
        Workspace workspace;

        // print all blobls
        std::cout << "current workspace has blobs" << std::endl;
        std::vector<std::string> blobs = workspace.Blobs();
        for(std::string &s:blobs){
            std::cout << s << std::endl;
        }

        //feed blob     
        std::vector<float> x(4*3*2);  //define blobs with std::vector
        int count = 0;
	    Tensor value = Tensor(x.size(), caffe2::DeviceType::CPU);
	    for (auto &v: x) {
		v = (float)rand() / RAND_MAX;
		value.mutable_data<float>()[count]=v;
		count++;
	    }

        std::cout << x << std::endl;

    	Blob* my_xBlob = workspace.CreateBlob("my_x");
		Tensor* tensor = BlobGetMutableTensor(my_xBlob, caffe2::DeviceType::CPU);

        tensor->ResizeLike(value);
        tensor->ShareData(value);

        // print all blobs
        std::cout << "current workspace has blobs" << std::endl;
        blobs = workspace.Blobs();
        for(std::string &s:blobs){
            std::cout << s << std::endl;
        }

        // fetch blob
    	Blob* blob = workspace.GetBlob("my_x");
        std::cout << "print the blob of my_x" << std::endl;
		print(blob, "my_x");

        // has blob
        std::cout << "current workspace has blob \"my_x\"?" << workspace.HasBlob("my_x") << std::endl;
    }
}
int main(int argc, char** argv) {
    caffe2::GlobalInit(&argc, &argv);
    caffe2::run();
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}