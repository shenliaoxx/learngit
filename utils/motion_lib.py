class MotionLibrary:
    """动作库，存储预定义的动作序列"""
    
    def __init__(self):
        self.motions = {
            "pinch": {
                "id": "pinch",
                "name": "捏取动作",
                "description": "将拇指与食指指尖相触，形成精细捏取姿势，适用于抓取小物体。",
                "duration": 5,  # 秒
                "video": "web_videos/pinch_20250314_122804_web.mp4",
                "key_points": ["保持手腕稳定", "只移动拇指和食指", "其他手指自然放松"],
                "collection": {
                    "repeat_times": 5,
                    "duration": 3,
                    "rest_time": 2,
                    "preparation_time": 3
                }
            },
            "grasp": {
                "id": "grasp",
                "name": "抓握动作",
                "description": "五指收拢，形成完整抓握姿势，适用于抓取和持物。",
                "duration": 5,
                "video": "web_videos/grasp_20250314_122825_web.mp4",
                "key_points": ["所有手指同时收拢", "保持适当力度", "手掌呈现自然弧度"],
                "collection": {
                    "repeat_times": 5,
                    "duration": 3,
                    "rest_time": 2,
                    "preparation_time": 3
                }
            },
            "flex": {
                "id": "flex",
                "name": "屈腕动作",
                "description": "手掌向下弯曲，形成屈腕姿势，适用于手腕灵活性训练。",
                "duration": 5,
                "video": "web_videos/flex_20250314_122903_web.mp4",
                "key_points": ["保持手指自然放松", "只移动手腕关节", "动作幅度适中"],
                "collection": {
                    "repeat_times": 5,
                    "duration": 3,
                    "rest_time": 2,
                    "preparation_time": 3
                }
            },
            "abduction_all_fingers": {
                "id": "abduction_all_fingers",
                "name": "所有手指外展",
                "description": "将所有手指向外展开，手掌保持平展，适用于手部灵活性训练。",
                "duration": 5,
                "video": "web_videos/abduction_all_fingers_20250318_165358_web.mp4",
                "key_points": ["手掌平展", "所有手指均匀向外展开", "保持手腕稳定", "避免手指过度用力"],
                "collection": {
                    "repeat_times": 5,
                    "duration": 3,
                    "rest_time": 2,
                    "preparation_time": 3
                }
            },
            "fist": {
                "id": "fist",
                "name": "握拳",
                "description": "将手指收拢，形成拳头，适用于力量训练和保护手部。",
                "duration": 5,
                "video": "web_videos/fist_20250318_165604_web.mp4",
                "key_points": ["保持手腕稳定", "手指完全收拢", "适当用力"],
                "collection": {
                    "repeat_times": 5,
                    "duration": 3,
                    "rest_time": 2,
                    "preparation_time": 3
                }
            },
            "thumbs_up": {
                "id": "thumbs_up",
                "name": "点赞动作",
                "description": "拇指向上，其余手指握拳，常用于表示赞同或鼓励。",
                "duration": 5,
                "video": "web_videos/thumbs_up_20250318_165659_web.mp4",
                "key_points": ["拇指挺直", "拳头握紧", "手腕自然"],
                "collection": {
                    "repeat_times": 5,
                    "duration": 3,
                    "rest_time": 2,
                    "preparation_time": 3
                }
            },
            "pointing": {
                "id": "pointing",
                "name": "指向动作",
                "description": "用食指指向目标或方向，常用于引导注意力。",
                "duration": 5,
                "video": "web_videos/pointing_20250318_165803_web.mp4",
                "key_points": ["手臂自然抬起", "食指伸直", "其他手指自然放松", "手腕保持稳定"],
                "collection": {
                    "repeat_times": 5,
                    "duration": 3,
                    "rest_time": 2,
                    "preparation_time": 3
                }
            },
            "large_grasp": {
                "id": "large_grasp",
                "name": "大范围抓握",
                "description": "用手掌包围较大的物体，适用于抓握大物体，如水瓶或球。",
                "duration": 5,
                "video": "web_videos/large_grasp_20250318_170324_web.mp4",
                "key_points": ["手掌完全展开", "手指均匀包围物体", "保持手腕稳定", "施加适当的抓握力度"],
                "collection": {
                    "repeat_times": 5,
                    "duration": 3,
                    "rest_time": 2,
                    "preparation_time": 3
                }
            },
            "writing_grasp": {
                "id": "writing_grasp",
                "name": "书写抓握",
                "description": "用手握住笔进行书写，适用于日常书写活动。",
                "duration": 5,
                "video": "web_videos/writing_grasp_20250318_170617_web.mp4",
                "key_points": ["拇指与食指形成夹持", "中指支撑笔杆", "手腕自然放松", "保持手指灵活以便书写"],
                "collection": {
                    "repeat_times": 5,
                    "duration": 3,
                    "rest_time": 2,
                    "preparation_time": 3
                }
            },
            
            "parallel_extension_grasp": {
                "id": "parallel_extension_grasp",
                "name": "平行伸展抓握",
                "description": "手掌和手指以平行伸展状态抓握书本或其他扁平物体。",
                "duration": 5,
                "video": "web_videos/parallel_extension_grasp_20250318_165129_web.mp4",
                "key_points": ["手指平行伸展", "手掌稳定", "书本水平", "手腕自然"],
                "collection": {
                    "repeat_times": 5,
                    "duration": 3,
                    "rest_time": 2,
                    "preparation_time": 3
                }
            }
        }

        # 预定义动作序列
        self.sequences = {
            "基础动作": ["pinch", "grasp", "flex", "abduction_all_fingers", "fist", "thumbs_up", "pointing"],
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