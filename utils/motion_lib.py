class MotionLibrary:
    """动作库，存储预定义的动作序列"""
    
    def __init__(self):
        self.motions = {
            "pinch": {
                "id": "pinch",
                "name": "捏取动作",
                "description": "将拇指与食指指尖相触，形成精细捏取姿势，适用于抓取小物体。",
                "duration": 5,  # 秒
                "video": "web_videos/捏取.mp4",
                "key_points": ["保持手腕稳定", "只移动拇指和食指", "其他手指自然放松"],
                "collection": {
                    "repeat_times": 6,
                    "duration": 4,
                    "rest_time": 4,
                    "preparation_time": 3
                }
            },
            "grasp": {
                "id": "grasp",
                "name": "抓握动作",
                "description": "五指收拢，形成完整抓握姿势，适用于抓取和持物。",
                "duration": 5,
                "video": "web_videos/抓握.mp4",
                "key_points": ["所有手指同时收拢", "保持适当力度", "手掌呈现自然弧度"],
                "collection": {
                    "repeat_times": 6,
                    "duration": 4,
                    "rest_time": 4,
                    "preparation_time": 3
                }
            },
            "thumb_index_shape": {
                "id": "thumb_index_shape",
                "name": "拇指-食指形状变换",
                "description": "拇指与食指交替形成圆形和三角形，训练手指的精细控制能力和形状感知。",
                "duration": 5,
                "video": "web_videos/对握姿势变换.mp4",
                "key_points": [
                    "保持其他手指自然放松",
                    "圆形：拇指和食指指尖轻触，形成完美圆形",
                    "三角形：拇指和食指指尖分开，形成等边三角形",
                    "确保形状转换流畅自然",
                    "保持动作节奏稳定"
                ],
                "collection": {
                    "repeat_times": 6,
                    "duration": 4,
                    "rest_time": 4,
                    "preparation_time": 3
                }
            },
            "abduction_all_fingers": {
                "id": "abduction_all_fingers",
                "name": "所有手指外展",
                "description": "将所有手指向外展开，手掌保持平展，适用于手部灵活性训练。",
                "duration": 5,
                "video": "web_videos/外展.mp4",
                "key_points": ["手掌平展", "所有手指均匀向外展开", "保持手腕稳定", "避免手指过度用力"],
                "collection": {
                    "repeat_times": 6,
                    "duration": 4,
                    "rest_time": 4,
                    "preparation_time": 3
                }
            },
            "fist": {
                "id": "fist",
                "name": "握拳",
                "description": "将手指收拢，形成拳头，适用于力量训练和保护手部。",
                "duration": 5,
                "video": "web_videos/握拳.mp4",
                "key_points": ["保持手腕稳定", "手指完全收拢", "适当用力"],
                "collection": {
                    "repeat_times": 6,
                    "duration": 4,
                    "rest_time": 4,
                    "preparation_time": 3
                }
            },
            "thumbs_up": {
                "id": "thumbs_up",
                "name": "点赞动作",
                "description": "拇指向上，其余手指握拳，常用于表示赞同或鼓励。",
                "duration": 5,
                "video": "web_videos/点赞.mp4",
                "key_points": ["拇指挺直", "拳头握紧", "手腕自然"],
                "collection": {
                    "repeat_times": 6,
                    "duration": 4,
                    "rest_time": 4,
                    "preparation_time": 3
                }
            },
            "pointing": {
                "id": "pointing",
                "name": "指向动作",
                "description": "用食指指向目标或方向，常用于引导注意力。",
                "duration": 5,
                "video": "web_videos/指向.mp4",
                "key_points": ["手臂自然抬起", "食指伸直", "其他手指自然放松", "手腕保持稳定"],
                "collection": {
                    "repeat_times": 6,
                    "duration": 4,
                    "rest_time": 4,
                    "preparation_time": 3
                }
            },
            "large_grasp": {
                "id": "large_grasp",
                "name": "大范围抓握",
                "description": "用手掌包围较大的物体，适用于抓握大物体，如水瓶或球。",
                "duration": 5,
                "video": "web_videos/握水杯.mp4",
                "key_points": ["手掌完全展开", "手指均匀包围物体", "保持手腕稳定", "施加适当的抓握力度"],
                "collection": {
                    "repeat_times": 6,
                    "duration": 5,
                    "rest_time": 5,
                    "preparation_time": 3
                }
            },
            "writing_grasp": {
                "id": "writing_grasp",
                "name": "书写抓握",
                "description": "用手握住笔进行书写，适用于日常书写活动。",
                "duration": 5,
                "video": "web_videos/握笔.mp4",
                "key_points": ["拇指与食指形成夹持", "中指支撑笔杆", "手腕自然放松", "保持手指灵活以便书写"],
                "collection": {
                    "repeat_times": 6,
                    "duration": 5,
                    "rest_time": 5,
                    "preparation_time": 3
                }
            },
            
            "parallel_extension_grasp": {
                "id": "parallel_extension_grasp",
                "name": "平行伸展抓握",
                "description": "手掌和手指以平行伸展状态抓握书本或其他扁平物体。",
                "duration": 5,
                "video": "web_videos/拿书.mp4",
                "key_points": ["手指平行伸展", "手掌稳定", "书本水平", "手腕自然"],
                "collection": {
                    "repeat_times": 6,
                    "duration": 5,
                    "rest_time": 5,
                    "preparation_time": 3
                }
            },
            "continuous_pinch": {
                "id": "continuous_pinch",
                "name": "连续对握",
                "description": "依次将食指、中指、无名指和小指分别与大拇指对握，训练手指的独立控制能力和协调性。",
                "duration": 5,
                "video": "web_videos/连续对握.mp4",
                "key_points": [
                    "保持手腕稳定",
                    "依次进行每根手指与大拇指的对握",
                    "确保每根手指独立完成动作",
                    "动作幅度适中",
                    "保持节奏均匀"
                ],
                "collection": {
                    "repeat_times": 6,
                    "duration": 4,
                    "rest_time": 4,
                    "preparation_time": 3
                }
            },
            "continuous_hook": {
                "id": "continuous_hook",
                "name": "连续勾手",
                "description": "依次将每根手指向掌心弯曲，形成连续勾手动作，训练手指的灵活性和独立控制能力。",
                "duration": 5,
                "video": "web_videos/连续勾手.mp4",
                "key_points": [
                    "保持手腕稳定",
                    "依次进行每根手指的弯曲",
                    "确保每根手指独立完成动作",
                    "动作幅度适中",
                    "保持节奏均匀",
                    "注意手指的协调性"
                ],
                "collection": {
                    "repeat_times": 6,
                    "duration": 4,
                    "rest_time": 4,
                    "preparation_time": 3
                }
            }
        }

        # 预定义动作序列
        self.sequences = {
            "基础动作": ["pinch", "grasp", "thumb_index_shape", "abduction_all_fingers", "fist", "thumbs_up", "pointing", "continuous_pinch", "continuous_hook"],
            "技能动作": ["large_grasp", "writing_grasp", "parallel_extension_grasp"],  # 示例技能动作
        }
    
    def get_motion(self, motion_id):
        """获取单个动作信息"""
        return self.motions.get(motion_id)
    
    def get_sequence(self, sequence_id):
        """获取动作序列"""
        sequence = self.sequences.get(sequence_id, [])
        return [self.get_motion(motion_id) for motion_id in sequence]
    
    def get_all_sequences(self):
        """获取所有可用的动作序列"""
        return {k: {"name": k, "motions": len(v)} for k, v in self.sequences.items()}