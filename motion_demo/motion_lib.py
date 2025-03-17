class MotionLibrary:
    """动作库，存储预定义的动作序列"""
    
    def __init__(self):
        self.motions = {
            "pinch": {
                "id": "pinch",
                "name": "捏取动作",
                "description": "将拇指与食指指尖相触，形成精细捏取姿势",
                "duration": 5,  # 秒
                "video": "videos/web_videos/pinch_20250314_122804_web.mp4",
                "key_points": ["保持手腕稳定", "只移动拇指和食指", "其他手指自然放松"],
                "collection": {
                    "repeat_times": 5,      # 需要重复采集的次数
                    "duration": 3,          # 每次采集持续时间（秒）
                    "rest_time": 2,         # 两次采集之间的休息时间（秒）
                    "preparation_time": 3   # 准备时间（秒）
                }
            },
            "grasp": {
                "id": "grasp",
                "name": "抓握动作",
                "description": "五指收拢，形成完整抓握姿势",
                "duration": 5,
                "video": "videos/web_videos/grasp_20250314_122825_web.mp4",
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
                "description": "手掌向下弯曲，形成屈腕姿势",
                "duration": 5,
                "video": "videos/web_videos/flex_20250314_122903_web.mp4",
                "key_points": ["保持手指自然放松", "只移动手腕关节", "动作幅度适中"],
                "collection": {
                    "repeat_times": 5,
                    "duration": 3,
                    "rest_time": 2,
                    "preparation_time": 3
                }
            },
            "extend": {
                "id": "extend",
                "name": "伸腕动作",
                "description": "手掌向上抬起，形成伸腕姿势",
                "duration": 5,
                "video": "videos/web_videos/extend_20250314_122926_web.mp4",
                "key_points": ["保持手指自然放松", "只移动手腕关节", "动作幅度适中"],
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
            "基础动作": ["pinch", "grasp", "flex", "extend"],
            "技能动作": ["抓握水杯", "握鼠标"],
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