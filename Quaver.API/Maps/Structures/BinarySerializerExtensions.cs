using System;
using System.Collections.Generic;
using System.IO;

namespace Quaver.API.Maps.Structures
{
    public static class BinarySerializerExtensions
    {
        public static void SerializeRaw<T>(this List<T> list, BinaryWriter writer) where T : IBinarySerializable<T>
        {
            foreach (var t in list)
            {
                t.Serialize(writer);
            }
        }
        public static void Serialize<T>(this List<T> list, BinaryWriter writer) where T : IBinarySerializable<T>
        {
            writer.Write(list.Count);
            list.SerializeRaw(writer);
        }
        public static void SerializeShort<T>(this List<T> list, BinaryWriter writer) where T : IBinarySerializable<T>
        {
            writer.Write((short)list.Count);
            list.SerializeRaw(writer);
        }
        public static void SerializeByte<T>(this List<T> list, BinaryWriter writer) where T : IBinarySerializable<T>
        {
            writer.Write((byte)list.Count);
            list.SerializeRaw(writer);
        }
        public static List<T> ReadRawList<T>(this BinaryReader reader, int count) where T : IBinarySerializable<T>
        {
            var res = new List<T>();
            for (var i = 0; i < count; i++)
            {
                var obj = Activator.CreateInstance<T>();
                obj.Parse(reader);
                res.Add(obj);
            }

            return res;
        }
        public static List<T> ReadList<T>(this BinaryReader reader) where T : IBinarySerializable<T>
        {
            var count = reader.ReadInt32();
            return reader.ReadRawList<T>(count);
        }
        public static List<T> ReadShortList<T>(this BinaryReader reader) where T : IBinarySerializable<T>
        {
            var count = reader.ReadInt16();
            return reader.ReadRawList<T>(count);
        }
        public static List<T> ReadByteList<T>(this BinaryReader reader) where T : IBinarySerializable<T>
        {
            var count = reader.ReadByte();
            return reader.ReadRawList<T>(count);
        }
    }
}