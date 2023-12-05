<script src="https://cdn.staticfile.org/axios/0.18.0/axios.min.js"></script>

<template>
  <el-container>
    <el-header>车牌检测识别系统</el-header>
    <el-main>
      <el-row>
        <el-col :span="12">
          <div class="grid-content bg-purple">
            <el-upload
              class="upload-demo"
              action="http://localhost:6006/file_rec"
              :on-preview="handlePreview"
              :on-remove="handleRemove"
              :file-list="fileList"
              list-type="picture">
              <el-button size="small" type="primary">点击上传</el-button>
              <div slot="tip" class="el-upload__tip">只能上传jpg/png文件，且不超过500kb</div>
            </el-upload>
            <el-button size="small" type="primary" @click="detectRecognize()">开始检测</el-button>
          </div>
        </el-col>
        <el-col :span="12">
          <div class="grid-content bg-purple-light">
            <div class="demo-fit" v-show="ifshow">
              <div class="block" v-for="fit in fits" :key="fit">
                  <el-avatar shape="square" :size="300" :fit="fit" :src="url"></el-avatar>
              </div>
              <el-row>
                <el-result icon="success" title="车牌号：" :subTitle="plate">
                  <template slot="extra">
                    <el-button type="primary" size="medium">下载检测结果</el-button>
                  </template>
                </el-result>
              </el-row>
            </div>
          </div>
        </el-col>
      </el-row>


    </el-main>
  </el-container>
</template>

<script>
const axios = require('axios');
export default {
    data() {
      return {
        fits: ['fill'],
        url: 'https://fuss10.elemecdn.com/e/5d/4a731a90594a4af544c0c25941171jpeg.jpeg',
        plate: "567",
        ifshow: false,
        fileList: []
      }
    },
		methods:{
      // upload(file, fileList) {
      //   console.log(file);
      // },
      // formatUrlFn() {
      //   // 获取文件名及后缀
      //   let index= this.backUrl.lastIndexOf(".");
      //   let extension=this.backUrl.substring(index+1,this.backUrl.length);//index是点的位置。点的位置加1再到结尾
      //   let fileName= this.backUrl.substring(0,index);
      //   // fileName.split("/")[fileName.split("/").length-1]
      //   console.log( fileName.split("/")[fileName.split("/").length-1],'--fileName--')
      //   console.log(extension,'--extension--')
      //   // 回显多张图片地址
      //   for(let i=0;i<this.imgUrl.length;i++){
      //     let bgUrl = {}
      //     bgUrl.url = this.imgUrl[i]
      //     this.fileList.push(bgUrl)
      //   }
      //   console.log('this.fileList', this.fileList)

      //   // 把检测结果渲染到结果展示区域
      //   axios.get("http://localhost:6006/file_rec").then(function (response) {
      //         console.log(response);
      //         console.log(response.data);//响应结果
              

      //     }).catch(function (error) {
      //         console.log(error);
      //     })

      // },
			detectRecognize() {
        let that = this
        axios.get('http://localhost:6006/detect_recognize').then((response) => {
            console.log(response);
            console.log(response.data);//响应结果
            let results = response.data
            console.log("显示响应结果")
            // 把检测结果渲染到结果展示区域
            that.url = results[0][0]
            that.plate = results[0][1]
            that.ifshow = true

        }).catch(function (error) {
            console.log(error);
        })     
      },
      handleRemove(file, fileList) {
        console.log(file, fileList);
      },
      handlePreview(file) {
        console.log(file);
      }
		},
    // mounted() {
    //   this.imgUrl = this.imgUrl.split(",")
    //   this.formatUrlFn()
    // },
  }

</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
  .el-header, .el-footer {
    background-color: #B3C0D1;
    color: #333;
    text-align: center;
    line-height: 60px;
  }
  
  .el-aside {
    background-color: #D3DCE6;
    color: #333;
    text-align: center;
    line-height: 200px;
  }
  
  .el-main {
    background-color: #E9EEF3;
    color: #333;
    text-align: center;
    line-height: 160px;
  }
  
  body > .el-container {
    margin-bottom: 40px;
  }
  
  .el-container:nth-child(5) .el-aside,
  .el-container:nth-child(6) .el-aside {
    line-height: 260px;
  }
  
  .el-container:nth-child(7) .el-aside {
    line-height: 320px;
  }
  .el-row {
    margin-bottom: 20px;
    &:last-child {
      margin-bottom: 0;
    }
  }
  .el-col {
    border-radius: 4px;
  }
  .bg-purple-dark {
    background: #99a9bf;
  }
  .bg-purple {
    background: #d3dce6;
  }
  .bg-purple-light {
    background: #e5e9f2;
  }
  .grid-content {
    border-radius: 4px;
    min-height: 36px;
  }
  .row-bg {
    padding: 10px 0;
    background-color: #f9fafc;
  }
</style>
