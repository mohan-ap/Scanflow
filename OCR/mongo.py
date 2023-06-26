from pymongo import MongoClient
from flask import jsonify

client = MongoClient('mongodb://localhost:27017')
db = client['qa']
files=db['files']
history = db['conversations']


def store_message(user_id, message):
    conversation = history.find_one({'user_id': user_id})
    if conversation:
        conversation['messages'].append(message)
        history.update_one({'user_id': user_id}, {'$set': {'messages': conversation['messages']}})
    else:
        conversation = {'user_id': user_id, 'messages': [message]}
        history.insert_one(conversation)

def retrieve_conversation(user_id):
    conversation = history.find_one({'user_id': user_id})
    if conversation:
        return conversation['messages']
    else:
      return jsonify({"message":"no history found"})
      