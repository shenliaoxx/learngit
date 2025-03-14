class MotionDemo {
    constructor() {
        // DOM元素
        this.sequenceSelect = document.getElementById('sequence-select');
        this.motionVideo = document.getElementById('motion-video');
        this.motionName = document.getElementById('motion-name');
        this.motionDescription = document.getElementById('motion-description');
        this.keyPointsList = document.getElementById('key-points-list');
        this.progressIndicator = document.getElementById('progress-indicator');
        this.sequenceSteps = document.getElementById('sequence-steps');
        this.prevBtn = document.getElementById('prev-btn');
        this.playBtn = document.getElementById('play-btn');
        this.nextBtn = document.getElementById('next-btn');
        this.statusMessage = document.getElementById('status-message');

        // 采集相关元素
        this.prepareCollectionBtn = document.getElementById('prepare-collection-btn');
        this.startCollectionBtn = document.getElementById('start-collection-btn');
        this.stopCollectionBtn = document.getElementById('stop-collection-btn');
        this.finishCollectionBtn = document.getElementById('finish-collection-btn');
        this.collectionOverlay = document.getElementById('collection-overlay');
        this.overlayStatus = document.getElementById('overlay-status');
        this.overlayMessage = document.getElementById('overlay-message');
        this.collectionProgress = document.getElementById('collection-progress');
        this.currentRepeat = document.getElementById('current-repeat');
        this.totalRepeats = document.getElementById('total-repeats');
        this.collectionStateElement = document.getElementById('collection-state');

        // 全屏倒计时元素
        this.fullscreenCountdown = document.getElementById('fullscreen-countdown');
        this.countdownTitle = document.getElementById('countdown-title');
        this.countdownTimer = document.getElementById('countdown-timer');
        this.countdownMessage = document.getElementById('countdown-message');

        // 状态变量
        this.sequences = {};
        this.currentSequence = null;
        this.currentMotionIndex = -1;
        this.currentMotion = null;
        this.timer = null;
        this.remainingTime = 0;
        this.isPlaying = false;

        // 采集状态
        this.collectionSession = null;
        this.collectionState = 'idle'; // idle, preparing, collecting, resting, completed
        this.collectionTimer = null;
        this.collectionConfig = null;

        // 视频事件监听
        this.motionVideo.addEventListener('ended', () => {
            this.stopMotion();
            this.updateStatus('视频播放完成');
        });

        // 初始化
        this.init();
    }

    async init() {
        // 加载所有动作序列
        await this.loadSequences();

        // 绑定事件
        this.bindEvents();

        this.updateStatus('系统就绪，请选择动作序列');
    }

    async loadSequences() {
        try {
            const response = await fetch('/api/sequences');
            const data = await response.json();

            if (data.status === 'success') {
                this.sequences = data.sequences;
                this.populateSequenceSelect();
            } else {
                console.error('加载动作序列失败:', data.message);
                this.updateStatus('加载动作序列失败', true);
            }
        } catch (error) {
            console.error('加载动作序列错误:', error);
            this.updateStatus('加载动作序列错误: ' + error.message, true);
        }
    }

    populateSequenceSelect() {
        // 清空现有选项
        this.sequenceSelect.innerHTML = '<option value="">-- 请选择 --</option>';

        // 添加序列选项
        for (const [id, sequence] of Object.entries(this.sequences)) {
            const option = document.createElement('option');
            option.value = id;
            option.textContent = `${sequence.name} (${sequence.motions}个动作)`;
            this.sequenceSelect.appendChild(option);
        }
    }

    async loadSequence(sequenceId) {
        try {
            this.updateStatus('正在加载动作序列...');

            const response = await fetch(`/api/sequence/${sequenceId}`);
            const data = await response.json();

            if (data.status === 'success') {
                this.currentSequence = data.sequence;
                this.currentMotionIndex = -1;
                this.updateSequenceProgress();
                this.nextMotion(); // 加载第一个动作
                this.updateStatus('动作序列已加载');

                // 重置采集状态
                this.resetCollectionState();
            } else {
                console.error('加载序列详情失败:', data.message);
                this.updateStatus('加载序列详情失败: ' + data.message, true);
            }
        } catch (error) {
            console.error('加载序列详情错误:', error);
            this.updateStatus('加载序列详情错误: ' + error.message, true);
        }
    }

    updateSequenceProgress() {
        // 清空步骤指示器
        this.sequenceSteps.innerHTML = '';

        // 创建步骤指示器
        if (this.currentSequence) {
            for (let i = 0; i < this.currentSequence.length; i++) {
                const step = document.createElement('div');
                step.className = 'step';
                step.textContent = this.currentSequence[i].name;
                step.title = this.currentSequence[i].description;
                this.sequenceSteps.appendChild(step);
            }

            // 更新进度条
            const progress = this.currentMotionIndex >= 0
                ? (this.currentMotionIndex + 1) / this.currentSequence.length * 100
                : 0;
            this.progressIndicator.style.width = `${progress}%`;

            // 更新步骤激活状态
            const steps = this.sequenceSteps.querySelectorAll('.step');
            steps.forEach((step, index) => {
                step.classList.toggle('active', index === this.currentMotionIndex);
            });
        }
    }

    loadMotion(motion) {
        if (!motion) return;

        this.currentMotion = motion;

        // 更新视频
        const videoPath = `/static/${motion.video}`;
        this.motionVideo.src = videoPath;
        this.motionVideo.load();

        // 更新信息
        this.motionName.textContent = motion.name;
        this.motionDescription.textContent = motion.description;

        // 更新关键点
        this.keyPointsList.innerHTML = '';
        motion.key_points.forEach(point => {
            const li = document.createElement('li');
            li.textContent = point;
            this.keyPointsList.appendChild(li);
        });

        // 更新倒计时
        this.remainingTime = motion.duration;

        // 更新按钮状态
        this.updateButtonStates();

        // 重置采集状态
        this.resetCollectionState();

        this.updateStatus(`已加载动作: ${motion.name}`);
    }

    nextMotion() {
        if (!this.currentSequence) return;

        // 停止当前动作
        this.stopMotion();

        // 移动到下一个动作
        this.currentMotionIndex++;
        if (this.currentMotionIndex >= this.currentSequence.length) {
            this.currentMotionIndex = this.currentSequence.length - 1;
            this.updateStatus('已到达序列末尾');
            return;
        }

        // 加载新动作
        this.loadMotion(this.currentSequence[this.currentMotionIndex]);
        this.updateSequenceProgress();
    }

    prevMotion() {
        if (!this.currentSequence) return;

        // 停止当前动作
        this.stopMotion();

        // 移动到上一个动作
        this.currentMotionIndex--;
        if (this.currentMotionIndex < 0) {
            this.currentMotionIndex = 0;
            this.updateStatus('已到达序列开头');
            return;
        }

        // 加载新动作
        this.loadMotion(this.currentSequence[this.currentMotionIndex]);
        this.updateSequenceProgress();
    }

    playMotion() {
        if (!this.currentMotion || this.isPlaying) return;

        this.isPlaying = true;
        this.motionVideo.play()
            .then(() => {
                // 开始倒计时
                this.startTimer();

                // 更新按钮状态
                this.playBtn.textContent = '暂停';
                this.updateButtonStates();

                this.updateStatus(`正在播放: ${this.currentMotion.name}`);
            })
            .catch(error => {
                console.error('视频播放失败:', error);
                this.isPlaying = false;
                this.updateStatus('视频播放失败: ' + error.message, true);
            });
    }

    pauseMotion() {
        if (!this.isPlaying) return;

        this.isPlaying = false;
        this.motionVideo.pause();

        // 停止倒计时
        this.stopTimer();

        // 更新按钮状态
        this.playBtn.textContent = '播放';
        this.updateButtonStates();

        this.updateStatus(`已暂停: ${this.currentMotion.name}`);
    }

    stopMotion() {
        this.pauseMotion();

        if (this.motionVideo) {
            this.motionVideo.currentTime = 0;
        }

        this.remainingTime = this.currentMotion ? this.currentMotion.duration : 0;
    }

    startTimer() {
        this.stopTimer();

        this.timer = setInterval(() => {
            this.remainingTime--;

            if (this.remainingTime <= 0) {
                this.stopTimer();
                this.motionComplete();
            }
        }, 1000);
    }

    stopTimer() {
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = null;
        }
    }

    motionComplete() {
        this.updateStatus(`动作完成: ${this.currentMotion.name}`);
    }

    updateButtonStates() {
        // 更新导航按钮状态
        this.prevBtn.disabled = !this.currentSequence || this.currentMotionIndex <= 0;
        this.nextBtn.disabled = !this.currentSequence ||
            this.currentMotionIndex >= this.currentSequence.length - 1;

        // 更新播放按钮状态
        this.playBtn.disabled = !this.currentMotion;

        // 更新采集按钮状态
        this.startCollectionBtn.disabled = !this.currentMotion;
    }

    updateStatus(message, isError = false) {
        this.statusMessage.textContent = message;
        this.statusMessage.style.color = isError ? '#e74c3c' : '#2c3e50';
    }

    bindEvents() {
        // 序列选择事件
        this.sequenceSelect.addEventListener('change', () => {
            const sequenceId = this.sequenceSelect.value;
            if (sequenceId) {
                this.loadSequence(sequenceId);
            } else {
                this.currentSequence = null;
                this.currentMotionIndex = -1;
                this.updateSequenceProgress();
                this.updateButtonStates();
                this.motionName.textContent = '请选择动作';
                this.motionDescription.textContent = '';
                this.keyPointsList.innerHTML = '';
                this.motionVideo.src = '';
                this.updateStatus('请选择动作序列');
            }
        });

        // 播放/暂停按钮事件
        this.playBtn.addEventListener('click', () => {
            if (this.isPlaying) {
                this.pauseMotion();
            } else {
                this.playMotion();
            }
        });

        // 上一个按钮事件
        this.prevBtn.addEventListener('click', () => {
            this.prevMotion();
        });

        // 下一个按钮事件
        this.nextBtn.addEventListener('click', () => {
            this.nextMotion();
        });

        // 准备采集按钮事件
        this.prepareCollectionBtn.addEventListener('click', () => {
            this.prepareCollection();
        });

        // 开始采集按钮事件
        this.startCollectionBtn.addEventListener('click', () => {
            this.startCollection();
        });

        // 停止采集按钮事件
        this.stopCollectionBtn.addEventListener('click', () => {
            this.stopCollection();
        });

        // 完成采集按钮事件
        this.finishCollectionBtn.addEventListener('click', () => {
            this.finishCollection();
        });

        // 视频点击事件 - 播放/暂停
        this.motionVideo.addEventListener('click', () => {
            if (this.isPlaying) {
                this.pauseMotion();
            } else {
                this.playMotion();
            }
        });

        // 键盘事件
        document.addEventListener('keydown', (event) => {
            switch (event.key) {
                case ' ':  // 空格键 - 播放/暂停
                    if (this.isPlaying) {
                        this.pauseMotion();
                    } else {
                        this.playMotion();
                    }
                    break;
                case 'ArrowLeft':  // 左箭头 - 上一个
                    this.prevMotion();
                    break;
                case 'ArrowRight':  // 右箭头 - 下一个
                    this.nextMotion();
                    break;
                case 'r':  // r键 - 重置当前视频
                    this.stopMotion();
                    break;
                case 'Escape':  // ESC键 - 关闭全屏倒计时
                    if (!this.fullscreenCountdown.classList.contains('hidden')) {
                        // 只有在倒计时显示时才处理
                        this.hideFullscreenCountdown();
                        // 如果有计时器，也停止它
                        if (this.collectionTimer) {
                            clearInterval(this.collectionTimer);
                            this.collectionTimer = null;
                        }
                        this.updateStatus('倒计时已手动取消');
                    }
                    break;
            }
        });

        // 点击全屏倒计时也可以关闭它
        this.fullscreenCountdown.addEventListener('click', () => {
            this.hideFullscreenCountdown();
            // 如果有计时器，也停止它
            if (this.collectionTimer) {
                clearInterval(this.collectionTimer);
                this.collectionTimer = null;
            }
            this.updateStatus('倒计时已手动取消');
        });
    }

    async prepareCollection() {
        if (!this.currentMotion) return;

        try {
            const response = await fetch('/api/collection/prepare', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    motion_id: this.currentMotion.id
                })
            });

            const data = await response.json();

            if (data.status === 'success') {
                this.collectionSession = data.session_id;
                this.collectionConfig = data.collection_config;
                this.collectionState = 'preparing';

                // 更新UI状态
                this.prepareCollectionBtn.disabled = true;
                this.startCollectionBtn.disabled = false;
                this.collectionProgress.classList.remove('hidden');
                this.collectionOverlay.classList.remove('hidden');

                // 显示准备倒计时
                this.overlayStatus.textContent = '准备采集';
                this.overlayMessage.textContent = '请准备好执行动作';

                // 更新采集状态显示
                this.collectionStateElement.textContent = '准备中';

                this.startPreparationCountdown(data.collection_config.preparation_time);

                this.updateStatus('准备采集中...');
            } else {
                throw new Error(data.message);
            }
        } catch (error) {
            console.error('准备采集失败:', error);
            this.updateStatus('准备采集失败: ' + error.message, true);
            this.resetCollectionState();
        }
    }

    async startCollection() {
        if (!this.collectionSession) return;

        try {
            const response = await fetch('/api/collection/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.collectionSession
                })
            });

            const data = await response.json();

            if (data.status === 'success') {
                this.collectionState = 'collecting';

                // 更新UI状态
                this.startCollectionBtn.disabled = true;
                this.stopCollectionBtn.disabled = false;
                this.overlayStatus.textContent = '采集进行中';
                this.overlayMessage.textContent = `第 ${data.repeat}/${data.total_repeats} 次采集`;

                // 更新采集状态显示
                this.collectionStateElement.textContent = '采集中';

                // 开始采集倒计时
                this.startCollectionCountdown(data.duration);

                this.updateStatus(`开始第 ${data.repeat}/${data.total_repeats} 次采集`);
            } else {
                throw new Error(data.message);
            }
        } catch (error) {
            console.error('开始采集失败:', error);
            this.updateStatus('开始采集失败: ' + error.message, true);
            this.resetCollectionState();
        }
    }

    async stopCollection() {
        if (!this.collectionSession) return;

        try {
            const response = await fetch('/api/collection/stop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.collectionSession
                })
            });

            const data = await response.json();

            if (data.status === 'success') {
                // 更新采集进度
                this.currentRepeat.textContent = data.repeat;
                this.totalRepeats.textContent = data.total_repeats;

                if (data.state === 'completed') {
                    // 采集完成
                    this.collectionState = 'completed';
                    this.overlayStatus.textContent = '采集完成';
                    this.overlayMessage.textContent = '所有采集已完成';
                    this.stopCollectionBtn.disabled = true;
                    this.finishCollectionBtn.disabled = false;

                    // 更新采集状态显示
                    this.collectionStateElement.textContent = '已完成';

                    this.updateStatus('采集完成');
                } else {
                    // 进入休息状态
                    this.collectionState = 'resting';
                    this.overlayStatus.textContent = '休息中';
                    this.overlayMessage.textContent = `休息 ${data.rest_time} 秒`;
                    this.stopCollectionBtn.disabled = true;
                    this.startCollectionBtn.disabled = false;

                    // 更新采集状态显示
                    this.collectionStateElement.textContent = '休息中';

                    this.startRestCountdown(data.rest_time);
                    this.updateStatus(`休息中: ${data.rest_time}秒`);
                }
            } else {
                throw new Error(data.message);
            }
        } catch (error) {
            console.error('停止采集失败:', error);
            this.updateStatus('停止采集失败: ' + error.message, true);
            this.resetCollectionState();
        }
    }

    async finishCollection() {
        if (!this.collectionSession) return;

        try {
            const response = await fetch('/api/collection/finish', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.collectionSession
                })
            });

            const data = await response.json();

            if (data.status === 'success') {
                // 重置采集状态
                this.resetCollectionState();

                // 隐藏采集相关UI
                this.collectionOverlay.classList.add('hidden');
                this.collectionProgress.classList.add('hidden');

                // 更新按钮状态
                this.prepareCollectionBtn.disabled = false;
                this.startCollectionBtn.disabled = true;
                this.stopCollectionBtn.disabled = true;
                this.finishCollectionBtn.disabled = true;

                this.updateStatus('采集会话已完成');
            } else {
                throw new Error(data.message);
            }
        } catch (error) {
            console.error('完成采集失败:', error);
            this.updateStatus('完成采集失败: ' + error.message, true);
            this.resetCollectionState();
        }
    }

    startPreparationCountdown(seconds) {
        // 显示全屏倒计时
        this.showFullscreenCountdown('准备采集', seconds, '请准备好执行动作');

        this.collectionTimer = setInterval(() => {
            seconds--;

            // 更新倒计时显示
            this.countdownTimer.textContent = seconds;

            if (seconds <= 0) {
                clearInterval(this.collectionTimer);
                // 隐藏全屏倒计时
                this.hideFullscreenCountdown();
                this.startCollection();
            }
        }, 1000);
    }

    startCollectionCountdown(seconds) {
        // 显示全屏倒计时
        this.showFullscreenCountdown('采集进行中', seconds, '请保持动作姿势');

        this.collectionTimer = setInterval(() => {
            seconds--;

            // 更新倒计时显示
            this.countdownTimer.textContent = seconds;

            if (seconds <= 0) {
                clearInterval(this.collectionTimer);
                // 隐藏全屏倒计时
                this.hideFullscreenCountdown();
                this.stopCollection();
            }
        }, 1000);
    }

    startRestCountdown(seconds) {
        // 显示全屏倒计时
        this.showFullscreenCountdown('休息中', seconds, '请放松，准备下一次采集');

        this.collectionTimer = setInterval(() => {
            seconds--;

            // 更新倒计时显示
            this.countdownTimer.textContent = seconds;

            if (seconds <= 0) {
                clearInterval(this.collectionTimer);
                // 隐藏全屏倒计时
                this.hideFullscreenCountdown();
                this.startCollection();
            }
        }, 1000);
    }

    // 显示全屏倒计时
    showFullscreenCountdown(title, seconds, message) {
        this.countdownTitle.textContent = title;
        this.countdownTimer.textContent = seconds;
        this.countdownMessage.textContent = message;
        this.fullscreenCountdown.classList.remove('hidden');
    }

    // 隐藏全屏倒计时
    hideFullscreenCountdown() {
        this.fullscreenCountdown.classList.add('hidden');
    }

    resetCollectionState() {
        // 重置采集状态
        this.collectionSession = null;
        this.collectionState = 'idle';
        this.collectionTimer = null;
        this.collectionConfig = null;

        // 清除定时器
        if (this.collectionTimer) {
            clearInterval(this.collectionTimer);
            this.collectionTimer = null;
        }

        // 重置UI状态
        this.prepareCollectionBtn.disabled = false;
        this.startCollectionBtn.disabled = true;
        this.stopCollectionBtn.disabled = true;
        this.finishCollectionBtn.disabled = true;
        this.collectionOverlay.classList.add('hidden');
        this.collectionProgress.classList.add('hidden');

        // 隐藏全屏倒计时
        this.hideFullscreenCountdown();

        // 重置采集状态显示
        if (this.currentRepeat) this.currentRepeat.textContent = '0';
        if (this.totalRepeats) this.totalRepeats.textContent = '5';
        if (this.collectionStateElement) this.collectionStateElement.textContent = '准备中';
    }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    new MotionDemo();
});