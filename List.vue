<template>
    <div>
        <h1>AI 模型預測</h1>
        <form @submit.prevent="submitData" enctype="multipart/form-data">
            <div>
                <label for="file">上傳照片或資料夾:</label>
                <input type="file" @change="handleFileUpload" multiple required>
            </div>
            <button type="submit">提交</button>
        </form>
        <div v-if="prediction">
            <h2>預測結果:</h2>
            <pre>{{ prediction.predicted_classes }}</pre>
            <h2>預測結果圖表:</h2>
            <img :src="prediction.chart" alt="Prediction Chart" />
        </div>
        <div v-if="defaultChart">
            <h2>預設圖表:</h2>
            <img :src="defaultChart" alt="Default Chart" />
        </div>
    </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';

const files = ref([]);
const prediction = ref(null);
const defaultChart = ref(null);

const handleFileUpload = (event) => {
    files.value = event.target.files;
};

const submitData = async () => {
    const formData = new FormData();
    for (let i = 0; i < files.value.length; i++) {
        formData.append('file', files.value[i]);
    }

    try {
        const response = await fetch('http://localhost:5000/api/analyze-file', {
            method: 'POST',
            body: formData,
            headers: {
                'Accept': 'application/json'
            }
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const result = await response.json();
        prediction.value = result;
    } catch (error) {
        console.error('Error:', error);
    }
};

// 加载默认图表
onMounted(async () => {
    try {
        const response = await fetch('http://localhost:5000/api/get-default-chart');
        const result = await response.json();
        defaultChart.value = result.chart;
    } catch (error) {
        console.error('Error fetching default chart:', error);
    }
});
</script>

<style scoped>
form {
    margin-bottom: 20px;
}

label {
    margin-right: 10px;
}

button {
    margin-top: 10px;
}
</style>