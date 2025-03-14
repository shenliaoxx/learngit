class MotionLibrary:
    """动作库，存储预定义的动作序列"""
    
    def __init__(self):
        self.motions = {
            "pinch": {
                "name": "捏取动作",
                "description": "将拇指与食指指尖相触，形成精细捏取姿势",
                "duration": 5,  # 秒
                "video": "/static/videos/web_videos/pinch_20250314_122804_web.mp4",
                "key_points": ["保持手腕稳定", "只移动拇指和食指", "其他手指自然放松"]
            },
            "grasp": {
                "name": "抓握动作",
                "description": "五指收拢，形成完整抓握姿势",
                "duration": 5,
                "video": "/static/videos/web_videos/grasp_20250314_122825_web.mp4",
                "key_points": ["所有手指同时收拢", "保持适当力度", "手掌呈现自然弧度"]
            },
            "flex": {
                "name": "屈腕动作",
                "description": "手掌向下弯曲，形成屈腕姿势",
                "duration": 5,
                "video": "/static/videos/web_videos/flex_20250314_122903_web.mp4",
                "key_points": ["保持手指自然放松", "只移动手腕关节", "动作幅度适中"]
            },
            "extend": {
                "name": "伸腕动作",
                "description": "手掌向上抬起，形成伸腕姿势",
                "duration": 5,
                "video": "/static/videos/web_videos/extend_20250314_122926_web.mp4",
                "key_points": ["保持手指自然放松", "只移动手腕关节", "动作幅度适中"]
            }
        }
        
        # 预定义动作序列
        self.sequences = {
            "basic_hand": {
                "name": "基础手部动作",
                "motions": ["pinch", "grasp", "flex", "extend"]
            },
            "fine_motor": {
                "name": "精细动作",
                "motions": ["pinch", "grasp"]
            },
            "wrist_motion": {
                "name": "手腕动作",
                "motions": ["flex", "extend"]
            }
        }
    
    def get_motion(self, motion_id):
        """获取单个动作信息"""
        return self.motions.get(motion_id)
    
    def get_sequence(self, sequence_id):
        """获取动作序列"""
        sequence = self.sequences.get(sequence_id)
        if not sequence:
            return None
        return {
            "name": sequence["name"],
            "motions": sequence["motions"]
        }
    
    def get_all_sequences(self):
        """获取所有可用的动作序列"""
        return {k: {"name": v["name"], "motions": len(v["motions"])} for k, v in self.sequences.items()} 