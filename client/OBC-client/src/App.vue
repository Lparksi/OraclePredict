<template>
  <div id="app">
    <!-- 头部导航 -->
    <header class="header">
      <div class="container">
        <h1 class="title">
          <span class="icon">🔮</span>
          Oracle Predict
        </h1>
        <p class="subtitle">甲骨文智能图像检测与分类系统</p>
      </div>
    </header>

    <!-- 主要内容区域 -->
    <main class="main">
      <div class="container">
        <!-- 状态指示器 -->
        <div class="status-bar">
          <div class="status-item" :class="{ active: apiStatus === 'healthy' }">
            <span class="status-dot" :class="apiStatus"></span>
            API状态: {{ apiStatusText }}
          </div>
          <div class="status-item">
            <span class="icon">📊</span>
            已处理: {{ processedCount }} 张图片
          </div>
        </div>

        <!-- 功能标签页 -->
        <div class="tabs">
          <button 
            class="tab-button" 
            :class="{ active: activeTab === 'single' }"
            @click="activeTab = 'single'"
          >
            单张预测
          </button>
          <button 
            class="tab-button" 
            :class="{ active: activeTab === 'batch' }"
            @click="activeTab = 'batch'"
          >
            批量预测
          </button>
          <button 
            class="tab-button" 
            :class="{ active: activeTab === 'info' }"
            @click="activeTab = 'info'; getModelInfo()"
          >
            模型信息
          </button>
        </div>

        <!-- 单张预测 -->
        <div v-if="activeTab === 'single'" class="tab-content">
          <div class="upload-section">
            <div class="upload-area" 
                 :class="{ 'drag-over': isDragging, 'has-image': selectedImage }"
                 @drop="handleDrop"
                 @dragover.prevent
                 @dragenter="isDragging = true"
                 @dragleave="isDragging = false">
              
              <div v-if="!selectedImage" class="upload-placeholder">
                <span class="upload-icon">📸</span>
                <p class="upload-text">拖拽图片到这里或点击选择</p>
                <p class="upload-hint">支持 JPG, PNG, BMP, WEBP 格式</p>
                <input type="file" 
                       ref="fileInput" 
                       class="file-input" 
                       accept="image/*"
                       @change="handleFileSelect">
                <button class="select-button" @click="triggerFileInput">
                  选择图片
                </button>
              </div>

              <div v-else class="image-preview">
                <img :src="selectedImage.url" :alt="selectedImage.name" class="preview-image">
                <div class="image-overlay">
                  <div class="image-info">
                    <p class="image-name">{{ selectedImage.name }}</p>
                    <p class="image-size">{{ selectedImage.size }}</p>
                  </div>
                  <button class="change-button" @click="clearImage">
                    更换图片
                  </button>
                </div>
              </div>
            </div>

            <div v-if="selectedImage" class="action-buttons">
              <button class="predict-button" 
                      :disabled="isLoading" 
                      @click="predictSingle">
                <span v-if="isLoading" class="loading-spinner"></span>
                {{ isLoading ? '分析中...' : '开始预测' }}
              </button>
              <button class="clear-button" @click="clearResults">
                清除结果
              </button>
            </div>
          </div>

          <!-- 预测结果 -->
          <div v-if="singleResult" class="results-section">
            <h3 class="results-title">预测结果</h3>
            <div class="result-stats">
              <div class="stat-item">
                <span class="stat-label">检测对象:</span>
                <span class="stat-value">{{ singleResult.count }}</span>
              </div>
              <div class="stat-item">
                <span class="stat-label">处理时间:</span>
                <span class="stat-value">{{ singleResult.processTime }}ms</span>
              </div>
            </div>
            
            <!-- 结果可视化 -->
            <div class="visualization-section">
              <h4 class="visualization-title">检测结果可视化</h4>
              <div class="result-image-container">
                <canvas 
                  ref="resultCanvas"
                  class="result-canvas"
                  @click="toggleVisualization"
                ></canvas>
                <div class="visualization-controls">
                  <button 
                    class="toggle-button" 
                    :class="{ active: showVisualization }"
                    @click="toggleVisualization"
                  >
                    {{ showVisualization ? '隐藏标注' : '显示标注' }}
                  </button>
                  <button class="download-button" @click="downloadResult">
                    下载结果图
                  </button>
                </div>
              </div>
            </div>
            
            <!-- 检测详情列表 -->
            <div class="detections">
              <h4 class="detections-title">检测详情</h4>
              <div v-for="(detection, index) in singleResult.results" 
                   :key="index" 
                   class="detection-item"
                   :class="{ highlighted: highlightedDetection === index }"
                   @mouseenter="highlightDetection(index)"
                   @mouseleave="highlightDetection(-1)">
                <div class="detection-info">
                  <span class="detection-class" :style="{ backgroundColor: getDetectionColor(index) }">
                    类别 {{ detection.class }}
                  </span>
                  <span class="detection-confidence">
                    置信度: {{ (detection.confidence * 100).toFixed(1) }}%
                  </span>
                </div>
                <div class="detection-bbox">
                  边界框: [{{ detection.bbox.join(', ') }}]
                </div>
                <div class="detection-size">
                  尺寸: {{ detection.bbox[2] - detection.bbox[0] }} × {{ detection.bbox[3] - detection.bbox[1] }}
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- 批量预测 -->
        <div v-if="activeTab === 'batch'" class="tab-content">
          <div class="batch-upload">
            <div class="batch-upload-area">
              <span class="upload-icon">📁</span>
              <p class="upload-text">批量上传图片</p>
              <input type="file" 
                     ref="batchFileInput" 
                     class="file-input" 
                     multiple
                     accept="image/*"
                     @change="handleBatchFileSelect">
              <button class="select-button" @click="triggerBatchFileInput">
                选择多张图片
              </button>
            </div>

            <div v-if="batchImages.length > 0" class="batch-preview">
              <div class="batch-header">
                <h3>已选择 {{ batchImages.length }} 张图片</h3>
                <div class="batch-actions">
                  <button class="predict-button" 
                          :disabled="isLoading" 
                          @click="predictBatch">
                    <span v-if="isLoading" class="loading-spinner"></span>
                    {{ isLoading ? `处理中 ${batchProgress}/${batchImages.length}...` : '批量预测' }}
                  </button>
                  <button class="clear-button" @click="clearBatchImages">
                    清除所有
                  </button>
                </div>
              </div>

              <div class="batch-images">
                <div v-for="(image, index) in batchImages" 
                     :key="index" 
                     class="batch-image-item">
                  <img :src="image.url" :alt="image.name" class="batch-thumbnail">
                  <div class="batch-image-info">
                    <p class="batch-image-name">{{ image.name }}</p>
                    <div v-if="batchResults[index]" class="batch-result">
                      <span v-if="batchResults[index].success" class="result-success">
                        ✅ 检测到 {{ batchResults[index].count }} 个对象
                      </span>
                      <span v-else class="result-error">
                        ❌ {{ batchResults[index].error }}
                      </span>
                    </div>
                  </div>
                  <button class="remove-button" @click="removeBatchImage(index)">
                    ✕
                  </button>
                </div>
              </div>
            </div>
          </div>

          <!-- 批量结果统计 -->
          <div v-if="batchResults.length > 0" class="batch-stats">
            <h3>批量处理统计</h3>
            <div class="stats-grid">
              <div class="stat-card">
                <div class="stat-number">{{ batchResults.length }}</div>
                <div class="stat-label">总处理数</div>
              </div>
              <div class="stat-card">
                <div class="stat-number">{{ successfulBatch }}</div>
                <div class="stat-label">成功</div>
              </div>
              <div class="stat-card">
                <div class="stat-number">{{ failedBatch }}</div>
                <div class="stat-label">失败</div>
              </div>
              <div class="stat-card">
                <div class="stat-number">{{ totalDetections }}</div>
                <div class="stat-label">总检测数</div>
              </div>
            </div>
          </div>
        </div>

        <!-- 模型信息 -->
        <div v-if="activeTab === 'info'" class="tab-content">
          <div v-if="modelInfo" class="model-info">
            <h3>模型配置信息</h3>
            <div class="info-grid">
              <div class="info-item">
                <span class="info-label">检测模型:</span>
                <span class="info-value">{{ modelInfo.detection_model }}</span>
              </div>
              <div class="info-item">
                <span class="info-label">分类模型:</span>
                <span class="info-value">{{ modelInfo.classification_model }}</span>
              </div>
              <div class="info-item">
                <span class="info-label">执行提供器:</span>
                <span class="info-value">{{ modelInfo.providers.join(', ') }}</span>
              </div>
              <div class="info-item">
                <span class="info-label">精度:</span>
                <span class="info-value">{{ modelInfo.precision }}</span>
              </div>
              <div class="info-item">
                <span class="info-label">置信度阈值:</span>
                <span class="info-value">{{ modelInfo.conf_threshold }}</span>
              </div>
              <div class="info-item">
                <span class="info-label">IoU阈值:</span>
                <span class="info-value">{{ modelInfo.iou_threshold }}</span>
              </div>
            </div>
          </div>
          
          <div v-if="!modelInfo && !isLoading" class="loading-placeholder">
            <p>点击模型信息标签页加载数据</p>
          </div>
        </div>
      </div>
    </main>

    <!-- 错误提示 -->
    <div v-if="errorMessage" class="error-toast" @click="clearError">
      <span class="error-icon">⚠️</span>
      {{ errorMessage }}
    </div>

    <!-- 成功提示 -->
    <div v-if="successMessage" class="success-toast" @click="clearSuccess">
      <span class="success-icon">✅</span>
      {{ successMessage }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'

// API 基础地址
const API_BASE_URL = 'http://localhost:5000'

// 模板引用
const fileInput = ref<HTMLInputElement>()
const batchFileInput = ref<HTMLInputElement>()
const resultCanvas = ref<HTMLCanvasElement>()

// 响应式数据
const activeTab = ref('single')
const isLoading = ref(false)
const isDragging = ref(false)
const apiStatus = ref('unknown')
const processedCount = ref(0)
const errorMessage = ref('')
const successMessage = ref('')

// 可视化相关
const showVisualization = ref(true)
const highlightedDetection = ref(-1)

// 单张预测相关
const selectedImage = ref<any>(null)
const singleResult = ref<any>(null)

// 批量预测相关
const batchImages = ref<any[]>([])
const batchResults = ref<any[]>([])
const batchProgress = ref(0)

// 模型信息
const modelInfo = ref<any>(null)

// 计算属性
const apiStatusText = computed(() => {
  switch (apiStatus.value) {
    case 'healthy': return '正常'
    case 'error': return '异常'
    case 'loading': return '检查中...'
    default: return '未知'
  }
})

const successfulBatch = computed(() => 
  batchResults.value.filter(r => r.success).length
)

const failedBatch = computed(() => 
  batchResults.value.filter(r => !r.success).length
)

const totalDetections = computed(() => 
  batchResults.value.reduce((total, r) => total + (r.success ? r.count : 0), 0)
)

// 生命周期钩子
onMounted(() => {
  checkApiHealth()
})

// API 调用函数
async function apiCall(endpoint: string, options: RequestInit = {}) {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      },
      ...options
    })
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }
    
    return await response.json()
  } catch (error: any) {
    throw new Error(error.message || '网络请求失败')
  }
}

// 检查API健康状态
async function checkApiHealth() {
  try {
    apiStatus.value = 'loading'
    await apiCall('/health')
    apiStatus.value = 'healthy'
  } catch (error: any) {
    apiStatus.value = 'error'
    showError(`API连接失败: ${error.message}`)
  }
}

// 文件处理函数
function triggerFileInput() {
  fileInput.value?.click()
}

function triggerBatchFileInput() {
  batchFileInput.value?.click()
}

function handleFileSelect(event: Event) {
  const target = event.target as HTMLInputElement
  if (target.files && target.files[0]) {
    const file = target.files[0]
    processFile(file)
  }
}

function handleDrop(event: DragEvent) {
  event.preventDefault()
  isDragging.value = false
  
  if (event.dataTransfer?.files && event.dataTransfer.files[0]) {
    const file = event.dataTransfer.files[0]
    processFile(file)
  }
}

function processFile(file: File) {
  if (!file.type.startsWith('image/')) {
    showError('请选择图片文件')
    return
  }

  const reader = new FileReader()
  reader.onload = (e) => {
    selectedImage.value = {
      name: file.name,
      size: formatFileSize(file.size),
      url: e.target?.result as string,
      file: file
    }
  }
  reader.readAsDataURL(file)
}

function handleBatchFileSelect(event: Event) {
  const target = event.target as HTMLInputElement
  if (target.files) {
    const files = Array.from(target.files)
    files.forEach(processFileForBatch)
  }
}

function processFileForBatch(file: File) {
  if (!file.type.startsWith('image/')) {
    showError(`${file.name} 不是图片文件`)
    return
  }

  const reader = new FileReader()
  reader.onload = (e) => {
    batchImages.value.push({
      name: file.name,
      size: formatFileSize(file.size),
      url: e.target?.result as string,
      file: file
    })
  }
  reader.readAsDataURL(file)
}

// 预测函数
async function predictSingle() {
  if (!selectedImage.value) {
    showError('请先选择图片')
    return
  }

  try {
    isLoading.value = true
    const startTime = Date.now()
    
    const formData = new FormData()
    formData.append('image', selectedImage.value.file)
    
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: formData
    })
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }
    
    const result = await response.json()
    const processTime = Date.now() - startTime
    
    singleResult.value = {
      ...result,
      processTime
    }
    
    processedCount.value++
    showSuccess('预测完成!')
    
    // 绘制结果到canvas
    await drawResultsOnCanvas()
    
  } catch (error: any) {
    showError(`预测失败: ${error.message}`)
  } finally {
    isLoading.value = false
  }
}

async function predictBatch() {
  if (batchImages.value.length === 0) {
    showError('请先选择图片')
    return
  }

  try {
    isLoading.value = true
    batchResults.value = []
    batchProgress.value = 0

    // 转换为base64格式进行批量预测
    const images = await Promise.all(
      batchImages.value.map(img => {
        return new Promise<string>((resolve) => {
          const reader = new FileReader()
          reader.onload = (e) => resolve(e.target?.result as string)
          reader.readAsDataURL(img.file)
        })
      })
    )

    const result = await apiCall('/predict/batch', {
      method: 'POST',
      body: JSON.stringify({ images })
    })

    batchResults.value = result.batch_results
    processedCount.value += result.total_processed
    batchProgress.value = batchImages.value.length
    
    showSuccess(`批量预测完成! 成功: ${successfulBatch.value}, 失败: ${failedBatch.value}`)
    
  } catch (error: any) {
    showError(`批量预测失败: ${error.message}`)
  } finally {
    isLoading.value = false
  }
}

async function getModelInfo() {
  try {
    isLoading.value = true
    const result = await apiCall('/model/info')
    modelInfo.value = result.model_info
  } catch (error: any) {
    showError(`获取模型信息失败: ${error.message}`)
  } finally {
    isLoading.value = false
  }
}

// 可视化函数
const detectionColors = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
  '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
]

function getDetectionColor(index: number): string {
  return detectionColors[index % detectionColors.length]
}

function highlightDetection(index: number) {
  highlightedDetection.value = index
  drawResultsOnCanvas()
}

async function drawResultsOnCanvas() {
  if (!resultCanvas.value || !selectedImage.value || !singleResult.value) return
  
  const canvas = resultCanvas.value
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  
  // 创建图片对象
  const img = new Image()
  
  return new Promise<void>((resolve) => {
    img.onload = () => {
      // 设置canvas尺寸
      const maxWidth = 800
      const maxHeight = 600
      let { width, height } = img
      
      // 计算缩放比例
      const scale = Math.min(maxWidth / width, maxHeight / height)
      width *= scale
      height *= scale
      
      canvas.width = width
      canvas.height = height
      
      // 绘制原图
      ctx.drawImage(img, 0, 0, width, height)
      
      // 如果显示可视化标注
      if (showVisualization.value) {
        drawDetections(ctx, width / img.width, height / img.height)
      }
      
      resolve()
    }
    
    img.src = selectedImage.value.url
  })
}

function drawDetections(ctx: CanvasRenderingContext2D, scaleX: number, scaleY: number) {
  if (!singleResult.value) return
  
  ctx.lineWidth = 3
  ctx.font = '16px Arial'
  ctx.textBaseline = 'top'
  
  singleResult.value.results.forEach((detection: any, index: number) => {
    const [x1, y1, x2, y2] = detection.bbox
    const color = getDetectionColor(index)
    const isHighlighted = highlightedDetection.value === index
    
    // 缩放坐标
    const scaledX1 = x1 * scaleX
    const scaledY1 = y1 * scaleY
    const scaledX2 = x2 * scaleX
    const scaledY2 = y2 * scaleY
    
    // 设置样式
    ctx.strokeStyle = color
    ctx.fillStyle = color
    
    // 如果被高亮，使用更粗的线条和半透明背景
    if (isHighlighted) {
      ctx.lineWidth = 5
      ctx.globalAlpha = 0.3
      ctx.fillRect(scaledX1, scaledY1, scaledX2 - scaledX1, scaledY2 - scaledY1)
      ctx.globalAlpha = 1
    }
    
    // 绘制边界框
    ctx.strokeRect(scaledX1, scaledY1, scaledX2 - scaledX1, scaledY2 - scaledY1)
    
    // 绘制标签背景
    const label = `类别 ${detection.class} (${(detection.confidence * 100).toFixed(1)}%)`
    const textMetrics = ctx.measureText(label)
    const textWidth = textMetrics.width + 16
    const textHeight = 24
    
    ctx.fillStyle = color
    ctx.fillRect(scaledX1, scaledY1 - textHeight, textWidth, textHeight)
    
    // 绘制标签文字
    ctx.fillStyle = 'white'
    ctx.fillText(label, scaledX1 + 8, scaledY1 - textHeight + 4)
    
    // 重置线条宽度
    ctx.lineWidth = 3
  })
}

function toggleVisualization() {
  showVisualization.value = !showVisualization.value
  drawResultsOnCanvas()
}

function downloadResult() {
  if (!resultCanvas.value) return
  
  const link = document.createElement('a')
  link.download = `prediction_result_${Date.now()}.png`
  link.href = resultCanvas.value.toDataURL()
  link.click()
}

// 清除函数
function clearImage() {
  selectedImage.value = null
  if (fileInput.value) {
    fileInput.value.value = ''
  }
}

function clearResults() {
  singleResult.value = null
  highlightedDetection.value = -1
  if (resultCanvas.value) {
    const ctx = resultCanvas.value.getContext('2d')
    ctx?.clearRect(0, 0, resultCanvas.value.width, resultCanvas.value.height)
  }
}

function clearBatchImages() {
  batchImages.value = []
  batchResults.value = []
  batchProgress.value = 0
  if (batchFileInput.value) {
    batchFileInput.value.value = ''
  }
}

function removeBatchImage(index: number) {
  batchImages.value.splice(index, 1)
  if (batchResults.value[index]) {
    batchResults.value.splice(index, 1)
  }
}

// 工具函数
function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

function showError(message: string) {
  errorMessage.value = message
  setTimeout(() => {
    errorMessage.value = ''
  }, 5000)
}

function showSuccess(message: string) {
  successMessage.value = message
  setTimeout(() => {
    successMessage.value = ''
  }, 3000)
}

function clearError() {
  errorMessage.value = ''
}

function clearSuccess() {
  successMessage.value = ''
}
</script>

<style scoped>
/* 全局样式 */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

#app {
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
}

/* 头部样式 */
.header {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.2);
  padding: 20px 0;
}

.title {
  color: white;
  font-size: 2.5rem;
  font-weight: bold;
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;
}

.subtitle {
  color: rgba(255, 255, 255, 0.8);
  font-size: 1.1rem;
  margin-left: 44px;
}

/* 主要内容样式 */
.main {
  padding: 40px 0;
}

/* 状态栏样式 */
.status-bar {
  background: rgba(255, 255, 255, 0.9);
  border-radius: 12px;
  padding: 16px 24px;
  margin-bottom: 30px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.status-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 500;
}

.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #ccc;
}

.status-dot.healthy {
  background: #10b981;
  box-shadow: 0 0 10px rgba(16, 185, 129, 0.4);
}

.status-dot.error {
  background: #ef4444;
  box-shadow: 0 0 10px rgba(239, 68, 68, 0.4);
}

/* 标签页样式 */
.tabs {
  display: flex;
  gap: 4px;
  margin-bottom: 30px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 4px;
}

.tab-button {
  flex: 1;
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  background: transparent;
  color: rgba(255, 255, 255, 0.7);
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
}

.tab-button:hover {
  color: white;
  background: rgba(255, 255, 255, 0.1);
}

.tab-button.active {
  background: white;
  color: #667eea;
  font-weight: 600;
}

/* 标签页内容样式 */
.tab-content {
  background: white;
  border-radius: 16px;
  padding: 32px;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
  min-height: 400px;
}

/* 上传区域样式 */
.upload-section {
  margin-bottom: 30px;
}

.upload-area {
  border: 3px dashed #cbd5e1;
  border-radius: 16px;
  padding: 40px;
  text-align: center;
  transition: all 0.3s ease;
  position: relative;
  min-height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.upload-area:hover {
  border-color: #667eea;
  background: rgba(102, 126, 234, 0.05);
}

.upload-area.drag-over {
  border-color: #667eea;
  background: rgba(102, 126, 234, 0.1);
  transform: scale(1.02);
}

.upload-area.has-image {
  border-style: solid;
  border-color: #10b981;
}

.upload-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
}

.upload-icon {
  font-size: 4rem;
  opacity: 0.6;
}

.upload-text {
  font-size: 1.25rem;
  font-weight: 600;
  color: #374151;
}

.upload-hint {
  color: #6b7280;
  font-size: 0.9rem;
}

.file-input {
  display: none;
}

.select-button {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  border: none;
  border-radius: 12px;
  padding: 14px 28px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 1rem;
}

.select-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
}

/* 图片预览样式 */
.image-preview {
  position: relative;
  width: 100%;
  max-width: 500px;
  margin: 0 auto;
}

.preview-image {
  width: 100%;
  height: auto;
  border-radius: 12px;
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
}

.image-overlay {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: linear-gradient(transparent, rgba(0, 0, 0, 0.8));
  color: white;
  padding: 20px;
  border-radius: 0 0 12px 12px;
  display: flex;
  justify-content: space-between;
  align-items: end;
}

.image-info {
  flex: 1;
}

.image-name {
  font-weight: 600;
  margin-bottom: 4px;
}

.image-size {
  font-size: 0.9rem;
  opacity: 0.8;
}

.change-button {
  background: rgba(255, 255, 255, 0.2);
  border: 1px solid rgba(255, 255, 255, 0.3);
  color: white;
  border-radius: 8px;
  padding: 8px 16px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.change-button:hover {
  background: rgba(255, 255, 255, 0.3);
}

/* 操作按钮样式 */
.action-buttons {
  display: flex;
  gap: 16px;
  justify-content: center;
  margin-top: 24px;
}

.predict-button {
  background: linear-gradient(135deg, #10b981, #059669);
  color: white;
  border: none;
  border-radius: 12px;
  padding: 14px 32px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 1rem;
}

.predict-button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(16, 185, 129, 0.4);
}

.predict-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

.clear-button {
  background: #6b7280;
  color: white;
  border: none;
  border-radius: 12px;
  padding: 14px 24px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.clear-button:hover {
  background: #4b5563;
  transform: translateY(-2px);
}

/* 加载动画 */
.loading-spinner {
  width: 16px;
  height: 16px;
  border: 2px solid transparent;
  border-top: 2px solid currentColor;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

/* 结果展示样式 */
.results-section {
  border-top: 1px solid #e5e7eb;
  padding-top: 30px;
}

.results-title {
  color: #111827;
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 20px;
}

/* 可视化区域样式 */
.visualization-section {
  margin-bottom: 30px;
  padding: 24px;
  background: #f8fafc;
  border-radius: 16px;
  border: 1px solid #e2e8f0;
}

.visualization-title {
  color: #1e293b;
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: 16px;
}

.result-image-container {
  position: relative;
  text-align: center;
}

.result-canvas {
  max-width: 100%;
  border-radius: 12px;
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
  cursor: pointer;
  transition: transform 0.3s ease;
}

.result-canvas:hover {
  transform: scale(1.02);
}

.visualization-controls {
  margin-top: 16px;
  display: flex;
  gap: 12px;
  justify-content: center;
}

.toggle-button {
  background: #6b7280;
  color: white;
  border: none;
  border-radius: 8px;
  padding: 8px 16px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
}

.toggle-button:hover {
  background: #4b5563;
}

.toggle-button.active {
  background: linear-gradient(135deg, #667eea, #764ba2);
}

.download-button {
  background: linear-gradient(135deg, #10b981, #059669);
  color: white;
  border: none;
  border-radius: 8px;
  padding: 8px 16px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
}

.download-button:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
}

.result-stats {
  display: flex;
  gap: 24px;
  margin-bottom: 24px;
  padding: 20px;
  background: #f3f4f6;
  border-radius: 12px;
}

.stat-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.stat-label {
  color: #6b7280;
  font-size: 0.9rem;
}

.stat-value {
  color: #111827;
  font-weight: 600;
  font-size: 1.1rem;
}

.detections {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.detections-title {
  color: #1e293b;
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: 16px;
}

.detection-item {
  background: #f9fafb;
  border: 2px solid #e5e7eb;
  border-radius: 12px;
  padding: 16px;
  transition: all 0.3s ease;
  cursor: pointer;
}

.detection-item:hover {
  background: #f3f4f6;
  border-color: #d1d5db;
  transform: translateY(-1px);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.detection-item.highlighted {
  background: #fef3c7;
  border-color: #f59e0b;
  box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);
}

.detection-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.detection-class {
  color: white;
  padding: 4px 12px;
  border-radius: 6px;
  font-size: 0.9rem;
  font-weight: 600;
  display: inline-block;
}

.detection-confidence {
  color: #059669;
  font-weight: 600;
}

.detection-bbox {
  color: #6b7280;
  font-size: 0.9rem;
  font-family: 'Courier New', monospace;
  margin-top: 8px;
}

.detection-size {
  color: #6b7280;
  font-size: 0.9rem;
  margin-top: 4px;
  font-style: italic;
}

/* 批量上传样式 */
.batch-upload-area {
  border: 3px dashed #cbd5e1;
  border-radius: 16px;
  padding: 40px;
  text-align: center;
  margin-bottom: 30px;
}

.batch-preview {
  margin-top: 30px;
}

.batch-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding-bottom: 16px;
  border-bottom: 1px solid #e5e7eb;
}

.batch-header h3 {
  color: #111827;
  font-size: 1.25rem;
}

.batch-actions {
  display: flex;
  gap: 12px;
}

.batch-images {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 16px;
  max-height: 400px;
  overflow-y: auto;
}

.batch-image-item {
  background: #f9fafb;
  border-radius: 12px;
  padding: 12px;
  position: relative;
  border: 1px solid #e5e7eb;
}

.batch-thumbnail {
  width: 100%;
  height: 120px;
  object-fit: cover;
  border-radius: 8px;
  margin-bottom: 8px;
}

.batch-image-name {
  font-size: 0.9rem;
  font-weight: 500;
  color: #374151;
  margin-bottom: 8px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.batch-result {
  font-size: 0.8rem;
  font-weight: 500;
}

.result-success {
  color: #059669;
}

.result-error {
  color: #dc2626;
}

.remove-button {
  position: absolute;
  top: 8px;
  right: 8px;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  border: none;
  background: rgba(220, 38, 38, 0.9);
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  transition: all 0.3s ease;
}

.remove-button:hover {
  background: #dc2626;
  transform: scale(1.1);
}

/* 批量统计样式 */
.batch-stats {
  margin-top: 30px;
  padding-top: 30px;
  border-top: 1px solid #e5e7eb;
}

.batch-stats h3 {
  color: #111827;
  font-size: 1.25rem;
  margin-bottom: 20px;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 16px;
}

.stat-card {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  padding: 20px;
  border-radius: 12px;
  text-align: center;
}

.stat-number {
  font-size: 2rem;
  font-weight: bold;
  margin-bottom: 4px;
}

.stat-label {
  font-size: 0.9rem;
  opacity: 0.9;
}

/* 模型信息样式 */
.model-info h3 {
  color: #111827;
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 24px;
}

.info-grid {
  display: grid;
  gap: 16px;
}

.info-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px;
  background: #f9fafb;
  border-radius: 8px;
  border: 1px solid #e5e7eb;
}

.info-label {
  font-weight: 600;
  color: #374151;
}

.info-value {
  color: #111827;
  font-family: 'Courier New', monospace;
  background: white;
  padding: 4px 8px;
  border-radius: 4px;
  border: 1px solid #d1d5db;
}

/* 提示消息样式 */
.error-toast, .success-toast {
  position: fixed;
  bottom: 20px;
  right: 20px;
  padding: 16px 24px;
  border-radius: 12px;
  color: white;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  z-index: 1000;
  animation: slideIn 0.3s ease;
  max-width: 400px;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
}

.error-toast {
  background: linear-gradient(135deg, #ef4444, #dc2626);
}

.success-toast {
  background: linear-gradient(135deg, #10b981, #059669);
}

@keyframes slideIn {
  from {
    transform: translateX(100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}

.loading-placeholder {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 200px;
  color: #6b7280;
  font-size: 1.1rem;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .container {
    padding: 0 16px;
  }
  
  .title {
    font-size: 2rem;
  }
  
  .subtitle {
    font-size: 1rem;
    margin-left: 36px;
  }
  
  .tabs {
    flex-direction: column;
  }
  
  .tab-content {
    padding: 20px;
  }
  
  .status-bar {
    flex-direction: column;
    gap: 12px;
    text-align: center;
  }
  
  .batch-header {
    flex-direction: column;
    gap: 16px;
    text-align: center;
  }
  
  .batch-actions {
    justify-content: center;
  }
  
  .result-stats {
    flex-direction: column;
    gap: 12px;
  }
  
  .action-buttons {
    flex-direction: column;
  }
  
  .stats-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .info-item {
    flex-direction: column;
    gap: 8px;
    text-align: center;
  }
  
  .batch-images {
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  }
  
  .visualization-section {
    padding: 16px;
  }
  
  .visualization-controls {
    flex-direction: column;
    gap: 8px;
  }
  
  .result-canvas {
    max-width: 100%;
  }
}

@media (max-width: 480px) {
  .stats-grid {
    grid-template-columns: 1fr;
  }
  
  .batch-images {
    grid-template-columns: 1fr;
  }
  
  .upload-area {
    padding: 20px;
    min-height: 200px;
  }
  
  .upload-icon {
    font-size: 3rem;
  }
  
  .upload-text {
    font-size: 1.1rem;
  }
}
</style>
